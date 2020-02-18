LOGO_STR = """
   ______                  _ _    __      __
  / ____/___  ____ _____  (_) |  / /___ _/ /
 / /   / __ \/ __ `/ __ \/ /| | / / __ `/ / 
/ /___/ /_/ / /_/ / / / / / | |/ / /_/ / /  
\____/\____/\__, /_/ /_/_/  |___/\__,_/_/   
           /____/                           
"""

WELCOME_MESSAGE_STR = """
Welcome to CogniVal! This tool allows you to evaluate word embeddings (predefined and custom) against the
cognitive sources described in CogniVal: A Framework for Cognitive Word Embedding Evaluation (Hollenstein et al., 2019).

The configuration 'demo' serves to demonstrate the functionality of this tool. In order to use it, open the general properties
of the configuration by executing:

> config open demo

If the error message "Window too small ..." appears, resize the terminal window. Remove the placeholder for the PATH property and it
will be automatically set to the CogniVal user directory (by default: $HOME/.cognival). Navigate the form with (Shift-)Tab and the
content of fields with the cursor keys and adjust e.g. the number of CPU cores to be used. 

Using the following command, you can show general properties of the configuration, cognitive sources and embeddings associated
with the configuration and per-source details (Note: not every embedding is necessarily associated with every cognitive source!). The demo configuration contains
only one cognitive source, eeg_zuco and one embedding type, GloVe embeddings with dimensionality 50, associated with a set of random embeddings
of matching dimensionality.

Note that CogniVal commands with multiple arguments always require that the arguments be given explicitely:

> config show configuration=demo details=True

CogniVal cognitive sources are installed by executing:

> install cognitive-sources 

GloVe embeddings are installed by executing (note that this installs GloVe embeddings for all dimensionalities, as they are provided in one archive):

> install embeddings glove.6B.50

A prompt will be shown, asking wether the user wants to perform a random embeddings comparison. Make sure to respond with "Yes" (default).
Note that random embedding generation greedily uses at most n-1 of n available CPU cores (up to 10 with default parametrization)
for parallelized generation. By default, ten sets of random embeddings are generated, for which results are later averaged
to improve robustness.

Random embeddings can be generated at any later point using the following command. Regeneration requires that force=True is added:

> install random-embeddings glove.6B.50

The experiment details (Word embedding specifics) of the eeg_zuco-glove combination can be edited using: 

> config experiment configuration=demo cognitive-sources=[eeg_zuco] embeddings=[glove.6B.50]

The changes are automatically propagated to the associated random embeddings. Note that if a cognitive source or embedding is not yet
part of the configuration, it is automatically populated from the reference configuration, which contains default parameters
for all cognitive sources and embeddings evaluated in the original CogniVal paper. Multiple experiments can be edited at once and if there
are multiple values for a field, this is indicated in the editor.

When entering a space after a command, a navigable list of subcommands or arguments along with default values (where applicable) is shown. Using Tab,
previous command parametrizations can be auto-completed. Cursor keys allow to navigate through the history of commands, analogous to e.g. Bash.

In order to execute to evaluation, both cognitive sources and embeddings need to be installed.

The experiments can be run with the following command:

> run demo

Finally, a HTML report can be generated and viewed in the default browser by executing:

> report configuration=demo open-html=True

Basic commands:
 - help: brief overview over all commands.
 - welcome: display this message using 'less'.
 - readme: display the README.md using 'less'.
 - clear: clear the interactive shell.
 - example-commands: Show a table of example calls along with descriptions
 - browse: Browse the user directory and view files in read-only mode (requires vim)
"""