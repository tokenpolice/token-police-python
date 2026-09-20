# Contributing

Thanks for helping make the TokenPolice SDK better.

## Found a bug?

Open an issue with the `token-police` version, your Python version, the LLM provider or
framework and its version, and a minimal reproduction. Leave out API keys and prompt text.

## Want to change something?

Pull requests are welcome. Small, focused changes land fastest.

1. Fork the repo and create a branch.
2. Make the change and add or update a test in `tests/`.
3. Check it:
   ```bash
   pip install -e ".[all,dev]"
   pytest
   ```
4. Open the PR and say what changed and why. We review every PR; accepted changes go into the
   next release, credited to you in the changelog.

## Security

Security issues go to security@tokenpolice.ai, not a public issue. See [SECURITY.md](SECURITY.md).

## License

By contributing you agree that your contribution is licensed under Apache-2.0, like the rest
of this repository.
