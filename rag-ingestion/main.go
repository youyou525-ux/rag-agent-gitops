package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/ledongthuc/pdf"
)

// --- 配置常量 ---
const (
	// Milvus配置
	milvusAddress  = "192.168.30.11:30530" // Milvus NodePort
	collectionName = "tech_docs"
	embeddingDim   = 768 // nomic-embed-text 模型维度

	// Ollama配置
	ollamaURL            = "http://192.168.30.11:32341" // Ollama NodePort
	ollamaEmbeddingModel = "nomic-embed-text"

	// 数据源路径（相对于 ~/rag-ingestion 目录）
	knowledgeBasePath = "../rag-agent-gitops/knowledge-base"
)

// --- 结构体定义 ---
type OllamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type OllamaEmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// --- 主函数 ---
func main() {
	log.Println("Starting RAG data ingestion pipeline...")
	ctx := context.Background()

	// 1. 连接Milvus
	milvusClient, err := client.NewClient(ctx, client.Config{Address: milvusAddress})
	if err != nil {
		log.Fatalf("Failed to connect to Milvus: %v", err)
	}
	defer milvusClient.Close()
	log.Println("Successfully connected to Milvus.")

	// 2. 检查并创建集合
	if err := setupMilvusCollection(ctx, milvusClient); err != nil {
		log.Fatalf("Failed to setup Milvus collection: %v", err)
	}

	// 3. 加载和切分文档
	docs, err := loadAndSplitDocuments(knowledgeBasePath)
	if err != nil {
		log.Fatalf("Failed to load and split documents: %v", err)
	}
	log.Printf("Loaded and split %d total document chunks.", len(docs))

	// 3a. 查询Milvus中已存在的文档ID
	existingIDs, err := queryExistingIDs(ctx, milvusClient)
	if err != nil {
		log.Fatalf("Failed to query existing IDs from Milvus: %v", err)
	}
	log.Printf("Found %d existing document IDs in Milvus.", len(existingIDs))

	// 4. 为新文档生成向量
	log.Println("Generating embeddings for new document chunks...")
	var newIDs []string
	var newTexts []string
	var newEmbeddings [][]float32

	for i, doc := range docs {
		docID := generateID(doc)
		if _, exists := existingIDs[docID]; exists {
			continue // 已存在则跳过
		}

		embedding, err := getOllamaEmbedding(doc)
		if err != nil {
			log.Printf("Failed to get embedding for chunk %d, skipping. Error: %v", i, err)
			continue
		}

		newIDs = append(newIDs, docID)
		newTexts = append(newTexts, doc)
		newEmbeddings = append(newEmbeddings, embedding)
	}
	log.Printf("Found %d new document chunks to insert.", len(newIDs))

	// 5. 将新数据插入Milvus
	if len(newIDs) > 0 {
		if err = insertDataToMilvus(ctx, milvusClient, newIDs, newTexts, newEmbeddings); err != nil {
			log.Fatalf("Failed to insert new data into Milvus: %v", err)
		}
	} else {
		log.Println("No new data to insert. Knowledge base is up-to-date.")
	}
	log.Println("Data ingestion pipeline finished successfully!")
}

// --- 辅助函数 ---

// 为文本生成唯一ID (SHA256哈希)
func generateID(text string) string {
	hash := sha256.Sum256([]byte(text))
	return hex.EncodeToString(hash[:])
}

// 查询Milvus中已存在的文档ID
func queryExistingIDs(ctx context.Context, c client.Client) (map[string]struct{}, error) {
	stats, err := c.GetCollectionStatistics(ctx, collectionName)
	if err != nil {
		return nil, fmt.Errorf("failed to get collection stats: %w", err)
	}
	rowCountStr, ok := stats["row_count"]
	if !ok || rowCountStr == "0" {
		log.Println("Collection is empty or stats not available.")
		return make(map[string]struct{}), nil
	}

	queryResult, err := c.Query(ctx, collectionName, []string{}, "", []string{"id"}, client.WithLimit(16383))
	if err != nil {
		return nil, fmt.Errorf("failed to query IDs: %w", err)
	}

	idSet := make(map[string]struct{})

	var idColumn entity.Column
	for _, col := range queryResult {
		if col.Name() == "id" {
			idColumn = col
			break
		}
	}

	if idColumn != nil {
		idVarCharColumn, ok := idColumn.(*entity.ColumnVarChar)
		if !ok {
			return nil, fmt.Errorf("ID column is not of type VarChar as expected")
		}

		for i := 0; i < idVarCharColumn.Len(); i++ {
			id, err := idVarCharColumn.ValueByIdx(i)
			if err != nil {
				log.Printf("Warning: could not get value for ID at index %d: %v", i, err)
				continue
			}
			idSet[id] = struct{}{}
		}
	} else {
		log.Println("Warning: ID column not found in query result.")
	}

	return idSet, nil
}

// 在Milvus中创建集合
func setupMilvusCollection(ctx context.Context, c client.Client) error {
	exists, err := c.HasCollection(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("failed to check for collection: %w", err)
	}
	if exists {
		log.Printf("Collection '%s' already exists. Loading it...", collectionName)
		return c.LoadCollection(ctx, collectionName, false)
	}

	log.Printf("Creating collection '%s'...", collectionName)
	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "Personal Knowledge Base",
		Fields: []*entity.Field{
			entity.NewField().WithName("id").WithDataType(entity.FieldTypeVarChar).WithMaxLength(64).WithIsPrimaryKey(true),
			entity.NewField().WithName("text").WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535),
			entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(embeddingDim),
		},
	}
	if err = c.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	log.Println("Creating index...")
	index, _ := entity.NewIndexIvfFlat(entity.L2, 128)
	if err = c.CreateIndex(ctx, collectionName, "embedding", index, false); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	log.Println("Loading collection...")
	return c.LoadCollection(ctx, collectionName, false)
}

// 加载并切分多种格式的文档
func loadAndSplitDocuments(path string) ([]string, error) {
	var allChunks []string
	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(1000),
		textsplitter.WithChunkOverlap(200),
	)

	err := filepath.Walk(path, func(filePath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		var textContent string
		log.Printf("Processing file: %s", filePath)

		switch filepath.Ext(filePath) {
		case ".md", ".txt":
			content, readErr := os.ReadFile(filePath)
			if readErr != nil {
				return readErr
			}
			textContent = string(content)

		case ".pdf":
			f, r, readErr := pdf.Open(filePath)
			if readErr != nil {
				log.Printf("Warning: could not open PDF %s: %v. Skipping.", filePath, readErr)
				return nil
			}
			defer f.Close()
			var buf bytes.Buffer
			b, readErr := r.GetPlainText()
			if readErr != nil {
				log.Printf("Warning: could not extract text from PDF %s: %v. Skipping.", filePath, readErr)
				return nil
			}
			buf.ReadFrom(b)
			textContent = buf.String()

		default:
			log.Printf("Skipping unsupported file type: %s", filePath)
			return nil
		}

		if textContent == "" {
			log.Printf("Warning: file %s is empty or content could not be extracted.", filePath)
			return nil
		}

		chunks, splitErr := splitter.SplitText(textContent)
		if splitErr != nil {
			return splitErr
		}

		for _, chunk := range chunks {
			if chunk != "" {
				allChunks = append(allChunks, chunk)
			}
		}
		return nil
	})
	return allChunks, err
}

// 调用Ollama API获取文本的Embedding
func getOllamaEmbedding(text string) ([]float32, error) {
	reqBody, _ := json.Marshal(OllamaEmbeddingRequest{Model: ollamaEmbeddingModel, Prompt: text})
	resp, err := http.Post(ollamaURL+"/api/embeddings", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama returned non-200 status: %d, body: %s", resp.StatusCode, string(body))
	}

	var ollamaResp OllamaEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, err
	}

	if len(ollamaResp.Embedding) != embeddingDim {
		return nil, fmt.Errorf("unexpected embedding dimension: got %d, want %d", len(ollamaResp.Embedding), embeddingDim)
	}

	return ollamaResp.Embedding, nil
}

// 将数据批量插入Milvus
func insertDataToMilvus(ctx context.Context, c client.Client, ids []string, texts []string, embeddings [][]float32) error {
	idCol := entity.NewColumnVarChar("id", ids)
	textCol := entity.NewColumnVarChar("text", texts)
	embeddingCol := entity.NewColumnFloatVector("embedding", embeddingDim, embeddings)

	if _, err := c.Insert(ctx, collectionName, "", idCol, textCol, embeddingCol); err != nil {
		return fmt.Errorf("failed to insert data: %w", err)
	}
	if err := c.Flush(ctx, collectionName, false); err != nil {
		return fmt.Errorf("failed to flush data: %w", err)
	}
	log.Printf("Successfully inserted and flushed %d records.", len(ids))
	return nil
}
