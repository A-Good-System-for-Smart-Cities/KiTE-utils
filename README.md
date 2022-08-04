# KiTE [![CI](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils/actions/workflows/CI.yml)
## (Summary)
- To format code, run `black .`
- To lint code, run `flake8 .`
- To install requirements, run `pip install -r requirements.txt`
- To run tests, run `pytest -s KiTE_utils`

---

## Contributing

1. Fork this Repo
2. Clone the Repo onto your computer -- You may need to setup an SSH Key on your device.
 - Run `pip install -r requirements.txt` to get all the packages you need.
3. Create a branch (`git checkout -b new-feature`)
4. Make Changes
5. Run necessary quality assurance tools
 - [Formatter](#Formatter), [Linter](#Linter) ,[Unit Tests](#Unit-Tests).
6. Add your changes (`git commit -am "Commit Message"` or `git add <whatever files/folders you want to add>` followed by `git commit -m "Commit Message"`)
7. Push your changes to the repo (`git push origin new-feature`)
8. Create a pull request

You can locally build the package with `pip install -e .` and run unit tests with `pytest -s KiTE_utils`.

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
2. After making changes, you can run `pytest -s KiTE_utils` to run all unit tests
