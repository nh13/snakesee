# Release Steps for snakesee

This file contains instructions for release steps that were deferred during initial development.
**Do not check this file into version control.**

## Prerequisites

Before releasing, ensure:
- [ ] All tests pass: `pixi run check`
- [ ] Coverage meets 95% threshold
- [ ] Documentation builds: `pixi run docs`
- [ ] CHANGELOG.md is updated with release notes

---

## 1. Make GitHub Repository Public

1. Go to https://github.com/nh13/snakesee/settings
2. Scroll to "Danger Zone" section
3. Click "Change visibility"
4. Select "Make public"
5. Confirm by typing the repository name

---

## 2. Configure PyPI Publishing

### 2.1 Create PyPI Account and API Token

1. Create account at https://pypi.org/account/register/
2. Go to Account Settings > API tokens
3. Create a new token with scope "Entire account" (or project-specific after first upload)
4. Copy the token (starts with `pypi-`)

### 2.2 Add PyPI Token to GitHub Secrets

1. Go to https://github.com/nh13/snakesee/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: paste your PyPI token
5. Click "Add secret"

### 2.3 Test Publishing to TestPyPI (Optional but Recommended)

1. Create account at https://test.pypi.org/account/register/
2. Create API token at TestPyPI
3. Add as GitHub secret named `TEST_PYPI_API_TOKEN`
4. Modify `.github/workflows/publish.yml` to publish to TestPyPI first:

```yaml
- name: Publish to TestPyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.TEST_PYPI_API_TOKEN }}
    repository-url: https://test.pypi.org/legacy/
```

5. Test installation: `pip install -i https://test.pypi.org/simple/ snakesee`

---

## 3. Create First Release

### 3.1 Update Version

1. Update version in `pyproject.toml`:
   ```toml
   version = "0.1.0"
   ```

2. Update version in `snakesee/__init__.py`:
   ```python
   __version__ = "0.1.0"
   ```

### 3.2 Create CHANGELOG.md

Create `CHANGELOG.md` in repository root:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - YYYY-MM-DD

### Added
- Initial release of snakesee
- Real-time TUI monitoring of Snakemake workflows
- Progress tracking with completion estimation
- Running and failed job displays
- Historical timing statistics per rule
- Vim-style keyboard navigation
- Multiple layout modes (full, compact, minimal)
- Job filtering by rule name
- Log file navigation for historical runs
- CLI commands: `snakesee watch` and `snakesee status`
```

### 3.3 Create Git Tag and GitHub Release

```bash
# Ensure you're on main branch with latest changes
git checkout main
git pull origin main

# Create annotated tag
git tag -a v0.1.0 -m "Release v0.1.0"

# Push tag to GitHub
git push origin v0.1.0
```

This will trigger the `publish.yml` workflow which will:
1. Build the package
2. Publish to PyPI
3. Create a GitHub release

### 3.4 Verify Release

1. Check PyPI: https://pypi.org/project/snakesee/
2. Check GitHub releases: https://github.com/nh13/snakesee/releases
3. Test installation: `pip install snakesee`

---

## 4. Publish to Bioconda

### 4.1 Prerequisites

- Package must be published to PyPI first
- You need a GitHub account
- Fork the bioconda-recipes repository

### 4.2 Create Bioconda Recipe

1. Fork https://github.com/bioconda/bioconda-recipes

2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/bioconda-recipes.git
   cd bioconda-recipes
   ```

3. Create recipe directory:
   ```bash
   mkdir -p recipes/snakesee
   ```

4. Create `recipes/snakesee/meta.yaml`:
   ```yaml
   {% set name = "snakesee" %}
   {% set version = "0.1.0" %}

   package:
     name: {{ name|lower }}
     version: {{ version }}

   source:
     url: https://pypi.io/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz
     sha256: <SHA256_HASH_FROM_PYPI>

   build:
     number: 0
     noarch: python
     script: {{ PYTHON }} -m pip install . -vv
     entry_points:
       - snakesee = snakesee.cli:main

   requirements:
     host:
       - python >=3.11
       - pip
       - hatchling
     run:
       - python >=3.11
       - rich >=13.0.0
       - defopt >=6.4.0

   test:
     imports:
       - snakesee
     commands:
       - snakesee --help

   about:
     home: https://github.com/nh13/snakesee
     license: MIT
     license_family: MIT
     license_file: LICENSE
     summary: A standalone TUI for monitoring Snakemake workflows
     description: |
       snakesee is a standalone terminal user interface (TUI) for monitoring
       Snakemake workflow progress. It provides real-time progress tracking,
       time estimation, and job status monitoring.
     dev_url: https://github.com/nh13/snakesee

   extra:
     recipe-maintainers:
       - YOUR_GITHUB_USERNAME
   ```

5. Get SHA256 hash from PyPI:
   ```bash
   curl -sL https://pypi.org/pypi/snakesee/json | python -c "import sys, json; print(json.load(sys.stdin)['urls'][0]['digests']['sha256'])"
   ```

### 4.3 Submit Pull Request

1. Commit your recipe:
   ```bash
   git checkout -b add-snakesee
   git add recipes/snakesee/
   git commit -m "Add snakesee recipe"
   git push origin add-snakesee
   ```

2. Open PR at https://github.com/bioconda/bioconda-recipes/compare

3. Wait for CI checks and maintainer review

4. Once merged, package will be available via:
   ```bash
   conda install -c bioconda snakesee
   ```

---

## 5. Set Up Documentation Hosting

### 5.1 Enable Read the Docs

1. Go to https://readthedocs.org/
2. Sign in with GitHub
3. Click "Import a Project"
4. Select `nh13/snakesee`
5. Configure:
   - Name: `snakesee`
   - Default branch: `main`
   - Documentation type: `Mkdocs`

The `.readthedocs.yml` file is already configured in the repository.

### 5.2 Verify Documentation

After enabling, documentation will be available at:
- https://snakesee.readthedocs.io/

---

## 6. Enable Codecov

1. Go to https://codecov.io/
2. Sign in with GitHub
3. Add the `nh13/snakesee` repository
4. Copy the Codecov token
5. Add as GitHub secret named `CODECOV_TOKEN`

The `codecov.yml` file and GitHub Actions workflow are already configured.

---

## Quick Reference: Release Checklist

For subsequent releases:

```bash
# 1. Update version in pyproject.toml and __init__.py
# 2. Update CHANGELOG.md
# 3. Commit changes
git add -A
git commit -m "Prepare release v0.x.x"
git push origin main

# 4. Create and push tag
git tag -a v0.x.x -m "Release v0.x.x"
git push origin v0.x.x

# 5. Update bioconda recipe (if needed)
# - Update version and sha256 in meta.yaml
# - Submit PR to bioconda-recipes
```
