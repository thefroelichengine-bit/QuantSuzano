# Documentation Assets

This folder contains documentation assets for the QuantSuzano project.

## Contents

### `/images/`
Contains all visualization images used in the main README.md:
- 12 professional-quality plots
- Generated from real market data (2020-2025)
- See [images/README.md](images/README.md) for details

## Usage in README

All images are referenced in the main README.md using relative paths:

```markdown
![Image Description](docs/images/image_name.png)
```

This ensures they display correctly on:
- GitHub
- GitLab
- Local markdown viewers
- Documentation sites

## Regenerating Images

```bash
# Run analysis to regenerate plots
python run_pipeline_safe.py all-robust

# Copy to docs folder
xcopy "data\out\plots\*.png" "docs\images\" /Y
```

## Notes

- Images are committed to Git for visibility
- Original plots are in `data/out/plots/` (gitignored)
- This folder (`docs/`) is NOT gitignored
- Keep synchronized with analysis updates

---

For more information, see the [main README](../README.md).

