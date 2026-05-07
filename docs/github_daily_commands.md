# GitHub Daily Push Commands

Run these commands every day after finishing work.

```bash
git status
git add .
git commit -m "day X: short description of completed work"
git push origin main
```

If this is your first push:

```bash
git init
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git add .
git commit -m "day 1: initialize project"
git push -u origin main
```

Good commit message examples:

```bash
git commit -m "day 2: add upload api and investigation status tracking"
git commit -m "day 3: add ai ocr and structured extraction"
git commit -m "day 4: add embeddings and pgvector chunk storage"
git commit -m "day 5: add keyword semantic and hybrid search"
git commit -m "day 6: add rag assistant and frontend"
git commit -m "day 7: finalize docs tests and docker setup"
```
