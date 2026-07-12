# Recipe: GitHub Actions gate

The one-liner (uses the published Action):

```yaml
# .github/workflows/rag-quality.yml
name: rag-quality
on: [pull_request]
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: abhay23-AI/raggate@v1
        with:
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}   # omit for heuristic mode
```

The score table is posted to the run **summary** automatically. Inputs:
`dir` (default `evals`), `gates`, `openai-api-key`, `python-version`, `version`.
