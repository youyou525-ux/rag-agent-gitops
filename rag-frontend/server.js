const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const port = 3000;
const goApiTarget = 'http://localhost:8080';

app.use('/api', createProxyMiddleware({
  target: goApiTarget,
  changeOrigin: true,
  pathRewrite: {
    '^/api': '',
  },
}));

app.use(express.static(path.join(__dirname, 'dist')));

app.use((req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(port, () => {
  console.log(`Frontend server with API proxy is listening at http://localhost:${port}`);
});
