# Go Best Practices

## Error Handling

In Go, it is idiomatic to handle errors by returning an `error` as the last return value. Always check them.

## Goroutine Management

Use `sync.WaitGroup` or channels to coordinate goroutines. Avoid goroutine leaks by always having a cancellation mechanism.

## Package Structure

Keep packages focused on a single responsibility. Internal packages should use the `internal/` directory convention.
