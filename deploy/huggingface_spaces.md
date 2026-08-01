# Deploying on Hugging Face Spaces

1. Create an account on huggingface.co, then **New Space**.
2. SDK: **Streamlit**. Visibility: **Private** if the content is sensitive.
3. Push the folder to the Space's repository:

```bash
git remote add hf https://huggingface.co/spaces/<username>/<space>
git push hf main
```

4. Add the following header **at the very top of README.md**, before
   everything else. Hugging Face reads it to configure the Space:

```
---
title: Quant Backtest Studio
emoji: "\u25e7"
colorFrom: gray
colorTo: yellow
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---
```

5. Password: **Settings -> Variables and secrets -> New secret**,
   name `password`.

The Space rebuilds on every `git push`.
