# Contributing to kelcode-ai-labs

First off, thanks for your interest in contributing! 🙏

## Getting Started

1. Fork the repo and clone your fork:
   ```bash
   git clone https://github.com/<your‑username>/kelcode-ai-labs.git
   cd kelcode-ai-labs
   ```
2. Create a branch for your change:
   ```bash
   git checkout -b feature/lab3-update-readme
   ```

## How to Submit Changes

- Run tests and linters before you push:
  ```bash
  pip install -r requirements.txt
  # run any lint/test commands here
  ```
- Commit your changes with a clear message:
  ```
  Lab 3: Add examples for quantisation parameters
  ```
- Push your branch and open a pull request against `main`.

## Code Style

- We use `black` and `flake8`. Please format your code with:
  ```bash
  black .
  flake8 .
  ```

## Reporting Issues

If you find a bug or want to request a new feature, please open an issue using the templates provided.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). By participating, you agree to abide by its terms.
```

Adding this file will make collaboration smoother and more professional—especially as your series grows and more folks start following along.

## Commit Message Style

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) spec.
Your commit messages should look like:

```
<type>(<scope>?): <description>

<body>?
<footer>?
```

Where **type** is one of:
- **feat**: a new feature
- **fix**: a bug fix
- **docs**: documentation only changes
- **style**: formatting, missing semicolons, etc; no code change
- **refactor**: code change that neither fixes a bug nor adds a feature
- **perf**: code change that improves performance
- **test**: adding missing tests or correcting existing ones
- **chore**: changes to the build process or auxiliary tools

**Example:**
```
feat(lab01): add disclaimer to README about synthetic data
fix(lab02): correct typo in starter prompt
docs(contributing): add Conventional Commit guidelines
```
```

That's all you need—no CI or enforcement required up front, but it sets clear expectations for anyone contributing.
