# SUAS-2022
Missouri S&amp;T Multirotor Robot Design Team's code for the Association for Unmanned Vehicle Systems International's 2022 Student Unmanned Aerial Systems Competition (AUVSI SUAS 2022)

## Table of Contents
- [Codebase Structure](#codebase-structure)
- [Requirements](#requirements)
- [Installation and Setup](#installation-and-setup)
- [Running / Testing Code](#running-and-testing-code)
- [Contributing](#contributing-code)
- [License](#license)

## Codebase Structure
```text
flight/  # Physical motor control, movement, path planning, and other flight related algorithms
vision/  # Mapping, shape & text detection, and other computer vision related algorithms
run.py  # Python program to run the competition code

integration_tests/  # Programs written to test discrete modules of the competition code

Information about files within each directory can be found in /<directory>/README.md
```

## Requirements
To run our competition code, you will need:
- A drone or drone [simulator](https://docs.px4.io/master/en/simulation/)
- Python version 3.8 or higher
- [Poetry](https://python-poetry.org/) (a list of managed dependencies can be found in the [project config](pyproject.toml))

## Installation and Setup
Follow these instructions exactly based on your platform:
1. Set up the development toolchain (platform dependent).
    - Follow the steps on [setting up your PX4 development toolchain](https://docs.px4.io/master/en/dev_setup/dev_env.html#setting-up-a-developer-environment-toolchain) based on your operating system

2. Install the proper Python version (see [Requirements](#requirements)) using pyenv on Linux and macOS. For Windows, get the executable from the [Python website](https://www.python.org/downloads/windows/).

3. [Install Poetry](https://python-poetry.org/docs/#installation))

4. Clone PX4 Firmware repository (tutorial [here](https://docs.px4.io/master/en/dev_setup/building_px4.html))

5. If testing without a drone, install a [supported simulator](https://docs.px4.io/master/en/dev_setup/dev_env.html#supported-targets)
    - Currently, we primarily do simple development with [jMAVSim](https://docs.px4.io/master/en/simulation/jmavsim.html), and complex development and testing in [AirSim](https://docs.px4.io/master/en/simulation/airsim.html), so start with jMAVSim
    - Run the `make` command from inside the PX4 Firmware repository

5. Clone the repository with `git clone --recursive https://github.com/MissouriMRR/SUAS-2022.git`

6. In the root of the repository, run `poetry install`

## Running and Testing Code
- Follow the steps in the [installation instructions](#Installation) above
- If you are working only on computer vision code, you may skip steps 1, 4, and 5
- Initialize your virtual environment for running and testing code with
```
poetry shell
```
- You may now run and test your modules at will inside this shell
- To run the competition code, execute the following from the root directory
```
./run.py
```
- When you're done, deactivate and exit the virtual env with
```
exit
```

## Contributing Code
1. Clone the repository with `git clone --recursive https://github.com/MissouriMRR/SUAS-2022.git`
2. Make sure you are on the most up-to-date `develop` branch with `git switch develop` then `git pull`
3. Create a branch with the proper naming format (see [Creating A Branch](#creating-a-branch))
4. Make changes in your branch and commit regularly (see [Committing Code](#committing-code))
5. Once changes are complete, push your code, go to GitHub, and submit a "Pull Request". Then, fill in the necessary information. Any issues your PR solves should be denoted in the description along with a note that the PR "Closes #XX" where XX is replaced with the issue number you are closing. Request a review from one of the software leaders on the upper  right-hand side of the PR.
6. Once it has been reviewed by the leads, it can be accepted and you can merge your branch into the develop branch.
7. Once this is done, you may delete your branch, and the cycle continues...

### Creating A Branch
- To contribute, you must create a new branch for your code
- If you've made changes in `develop` and want to add your changes to a new branch, use
```
git switch -c "branch_name"
```
- Otherwise, create & switch to a new branch with 
```
git checkout -b "branch_name" 
# or both of 
git branch "branch_name"
git checkout "branch_name"
```
   > `branch_name` should follow the convention `feature/{feature_name}`, or `hotfix/{fix_name}`
### Committing Code
- When programming with a VCS like Github, you should make changes and commit regularly with clear commit messages.
- This repo uses git hooks for the automated execution of scrips that reformat & analyze your code when a commit is made.
- In order for a pull request to be considered, commits must be made on a repo clone with pre-commit properly installed
#### Setting Up Pre-Commit
- Here are the commands for the first-time setup & install of pre-commit for the repo
```
pip install pre-commit
pre-commit install
```
- You can then use `pre-commit run` to run the pre-commit hooks on all staged files or you can let it automatically trigger when you make a commit
- Note that if you decide not to use `pre-commit run` before committing code, and your code gets reformatted but passes all of the analysis hooks, you must re-commit that code with a new second commit message either mirroring the previous commit message or stating that the code was reformatted. 
#### Committing Unfinished Code
- When committing code, our pre-commit hooks will scan your files and look for errors, type inconsistencies, bad practices, and non-conformities to our coding style.
- This means that your commit will be rejected if it fails any one of the pre-commit hooks.
- Oftentimes, one may need to commit code to save their place or the current version in unfinished code and bypass pre-commit hooks
##### Bypassing All Pre-Commit Hooks
- To bypass all pre-commit hooks, add the `--no-verify` flag to your `git commit` execution.
##### Bypassing Invidual Pre-Commit Hooks
- To bypass specific hooks on a module, place the following comments at the beginning of your file(s) for each respective hook.
    - Black: `# fmt: off`
    - Mypy: `# type: ignore`
    - Pylint: `# pylint: disable=all`
- Pull requests made that bypass pre-commit hooks without prior approval will be rejected.

## License
We adopt the MIT License for our projects. Please read the [LICENSE](LICENSE) file for more info
