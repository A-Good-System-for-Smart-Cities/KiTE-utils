# KiTE [![CI](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils/actions/workflows/CI.yml) [![Documentation Test](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils/actions/workflows/docs.yml/badge.svg)](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils/actions/workflows/docs.yml) [![.github/workflows/update-docs.yml](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils/actions/workflows/update-docs.yml/badge.svg)](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils/actions/workflows/update-docs.yml)

![](doc_template/logo-kite.jpg)


## (Summary)
- To format code, run `black .`
- To lint code, run `flake8 .`
- To install requirements, run `pip install -r requirements.txt`
- To run tests, run `pytest -s KiTE`

---

## Contributing

1. Fork this Repo
2. Clone the Repo onto your computer -- You may need to setup an SSH Key on your device.
 - Run `pip install -r requirements.txt` to get all the packages you need.
3. Create a branch (`git checkout -b new-feature`)
4. Make Changes
5. Run necessary quality assurance tools
 - [Formatter](#Formatter), [Linter](#Linter), [Unit Tests](#Unit-Tests).
6. Add your changes (`git commit -am "Commit Message"` or `git add <whatever files/folders you want to add>` followed by `git commit -m "Commit Message"`)
7. Push your changes to the repo (`git push origin new-feature`)
8. Create a pull request

You can locally build the package with `pip install -e .` and run unit tests with `pytest -s KiTE`.

---
## Code Quality Tools
### [black formatter](https://github.com/psf/black) automatically formats code

1. Run `pip install black` to get the package.
2. After making changes, run `black ./`.

### [flake8](https://github.com/pycqa/flake8) lints code
> Notifies you if any code is not currently up to Python standards.

1. Run `pip install flake8` to get the package.
2. After making changes, run `flake8`.

### [pytest](https://github.com/pytest-dev/pytest) runs unit tests

1. Run `pip install pytest` to get the package.
2. After making changes, you can run `pytest -s KiTE` to run all unit tests

---
## Update Documentation
We use [pdoc](https://github.com/mitmproxy/pdoc) to create KiTE's documentation.
1. `pip install pdoc` into your working environment
2. Preview edits with `pdoc -t doc_template --docformat numpy KiTE`.
    * This will locally host the documentation site (that will update as you make edits)
3. Edit documentation by adding/modifying docstrings in `KiTE`
