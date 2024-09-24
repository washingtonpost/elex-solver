# Changelog
### 2.1.1 - 2024-09-12
- chore: Updated all required packages to their latest versions.  Switched `QuantileRegressionSolver`'s solver from `ECOS` to `Clarabel` [#22](https://github.com/washingtonpost/elex-solver/pull/22)	

### 2.1.0 - 2024-03-29
- feat: Transition matrix solver. [#17](https://github.com/washingtonpost/elex-solver/pull/17)

### 2.0.1 - 2023-10-20
- chore: Updated all required packages to their latest versions.  Added `numpy` as a requirement. [#14](https://github.com/washingtonpost/elex-solver/pull/14)

### 2.0.0 - 2023-09-22
- feat: adding ordinary least squares regression solver. Updating quantile regression to solve dual [#11](https://github.com/washingtonpost/elex-solver/pull/11)

### 1.1.0 - 2023-04-21
- fix: Not regularizing intercept coefficient + better warning handling [#8](https://github.com/washingtonpost/elex-solver/pull/8)
- feat: Throw error when encountering NaN/Inf [#7](https://github.com/washingtonpost/elex-solver/pull/7)
- fix: fix deprecated warning [#6](https://github.com/washingtonpost/elex-solver/pull/6)
- chore: Add pre-commit linting and hook [#5](https://github.com/washingtonpost/elex-solver/pull/5)
- feat: Add regularization [#4](https://github.com/washingtonpost/elex-solver/pull/4)

### 1.0.3 - 2022-11-07
 - Add gitignore, codeowners, PR template, unit test workflow

### 1.0.2 - 2022-09-14
 - Specify README.md file type

### 1.0.1 - 2022-09-12
 - Update codeowners to public news engineering group

### 1.0.0 - 2022-09-01
 - Remove CircleCI and general clean-up
 - Add installation and contribution instructions for open source

### 0.0.3 - 2022-03-28
 - Check condition of design matrix

### 0.0.2 - 2022-02-17
 - Ability to save problems
 - Normalizing weights

### Initial release - 2022-01-24
