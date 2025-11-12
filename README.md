# aind-dynamic-foraging-basic-analysis

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.9-blue?logo=python)



## Usage
 - To use this template, click the green `Use this template` button and `Create new repository`.
 - After github initially creates the new repository, please wait an extra minute for the initialization scripts to finish organizing the repo.
 - To enable the automatic semantic version increments: in the repository go to `Settings` and `Collaborators and teams`. Click the green `Add people` button. Add `svc-aindscicomp` as an admin. Modify the file in `.github/workflows/tag_and_publish.yml` and remove the if statement in line 10. The semantic version will now be incremented every time a code is committed into the main branch.
 - To publish to PyPI, enable semantic versioning and uncomment the publish block in `.github/workflows/tag_and_publish.yml`. The code will now be published to PyPI every time the code is committed into the main branch.
 - The `.github/workflows/test_and_lint.yml` file will run automated tests and style checks every time a Pull Request is opened. If the checks are undesired, the `test_and_lint.yml` can be deleted. The strictness of the code coverage level, etc., can be modified by altering the configurations in the `pyproject.toml` file and the `.flake8` file.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

## Usage

Example notebooks to test and try different functions of this package (and the data-utils) package can be found here: 

https://codeocean.allenneuraldynamics.org/capsule/6488619/tree

Please duplicate the capsule to test the different functions. 


### Annotate licks
To create a dataframe of licks that has been annotated with licking bout starts/stops, cue responsive licks, reward triggered licks, and intertrial choices.
```
import aind_dynamic_foraging_basic_analysis.licks.annotation as annotation
df_licks = annotation.annotate_licks(nwb)
```

You can then plot interlick interval analyses with:
```
import aind_dynamic_foraging_basic_analysis.licks.plot_interlick_interval as pii

#Plot interlick interval of all licks
pii.plot_interlick_interval(df_licks)

#plot interlick interval for left and right licks separately
pii.plot_interlick_interval(df_licks, categories='event')
```

### Create lick analysis report
To create a figure with several licking pattern analyses:

```
import aind_dynamic_foraging_basic_analysis.licks.lick_analysis as lick_analysis
lick_analysis.plot_lick_analysis(nwb)
```

### Compute trial by trial metrics
To annotate the trials dataframe with trial by trial metrics:

```
import aind_dynamic_foraging_basic_analysis.metrics.trial_metrics as tm
df_trials = tm.compute_all_trial_metrics(nwb)
```

### Plot interactive session scroller
```
import aind_dynamic_foraging_basic_analysis.plot.plot_session_scroller as pss
pss.plot_session_scroller(nwb)
```

To disable lick bout and other annotations:
```
pss.plot_session_scroller(nwb,plot_bouts=False)
```

This function will automatically plot FIP data if available. To change the processing method plotted use:
```
pss.plot_session_scroller(nwb, processing="bright")
```

To change which trial by trial metrics plotted:
```
pss.plot_session_scroller(nwb, metrics=['response_rate'])
```

### Plot FIP PSTH
You can use the `plot_fip` module to compute and plot PSTHs for the FIP data. 

To compare one channel to multiple event types
```
from aind_dynamic_foraging_basic_analysis.plot import plot_fip as pf
channel = 'G_1_dff-poly'
rewarded_go_cues = nwb.df_trials.query('earned_reward == 1')['goCue_start_time_in_session'].values
unrewarded_go_cues = nwb.df_trials.query('earned_reward == 0')['goCue_start_time_in_session'].values
pf.plot_fip_psth_compare_alignments(
    nwb, 
    {'rewarded goCue':rewarded_go_cues,'unrewarded goCue':unrewarded_go_cues}, 
    channel, 
    censor=True
    )
```

To compare multiple channels to the same event type:
```
pf.plot_fip_psth(nwb, 'goCue_start_time')
```

### Hierarchical Bootstrapping and significance testing
If you want to perform hierarchical bootstrapping, then you can select `hb_sem` as the error type
```
fig, ax, etrs = plot_fip_psth_compare_alignments(
    nwbs, 
    alignments, 
    channel_name, 
    error_type='hb_sem',
    hierarchical_params={'nboots':10000}
)
```
Then add significance with:
```
fip_psth_stats_plot(ax, etrs['stats'])`, color='k', threshold=0.05)
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).
