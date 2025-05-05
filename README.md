# MT Exercise 3: Layer Normalization for Transformer Models

This repo is a collection of scripts showing how to install [JoeyNMT](https://github.com/joeynmt/joeynmt), download
data and train & evaluate models, as well as the necessary data for training your own model

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3.10 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps for macOS & Linux users

Clone this repository or your fork thereof in the desired place:

    git clone https://github.com/marpng/mt-exercise-03

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Make sure to install the exact software versions specified in the the exercise sheet before continuing.

Download Moses for post-processing:

    ./scripts/download_install_packages.sh


Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved. It is also possible to continue training from there later on.

# Steps for Windows users

This repo relies on Bash scripts (.sh files), which do not run natively on Windows (CMD or PowerShell).  
Here are two ways to make it work:

Option 1: Use WSL (Windows Subsystem for Linux)
Enable WSL and install Ubuntu: `wsl --install`

Open Ubuntu from your Start menu.

Inside the Ubuntu terminal, follow the exact same steps as shown above for macOS/Linux:
```
git clone https://github.com/marpng/mt-exercise-4
cd mt-exercise-4
./scripts/make_virtualenv.sh
./scripts/download_install_packages.sh
./scripts/train.sh
```     

Option 2: Manually run steps without shell scripts
If you can't use WSL, you can recreate the process manually using PowerShell or CMD
Create and activate a virtual environment:
```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
Manually download and install Moses and other dependencies (you'll need to look inside scripts/download_install_packages.sh to replicate its steps).

Run the training logic by manually executing the code inside train.sh, or porting it to a Python script or notebook.


# Task 1: Understanding Code: LayerNorm in JoeyNMT

In transformer_layers.py we can see that the TransformerEncoderLayer and TransformerDecoderLayer are set to layer_norm="post" in their constructor definitions. This means that if no configuration is provided, the model will apply post-norm by default.

    ../joeynmt/scripts/transformer_layers.py 


``` python
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        alpha: float = 1.0,
        layer_norm: str = "post", # here we see the default value which is post-normalization
        activation: str = "relu",
    ) -> None:
```

However, we noticed that JoeyNMT supports both pre-norm and post-norm. We can see this well in the forward pass of the encoder and decoder layers.

``` python
 def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        residual = x
        if self._layer_norm_position == "pre": # Here it executes LayerNorm before applying masked attention/dropout/residuals
            x = self.layer_norm(x)

        x, _ = self.src_src_att(x, x, x, mask)
        x = self.dropout(x) + self.alpha * residual

        if self._layer_norm_position == "post": # Here the masked attention/dropout/residuals are executed already and now post LayerNorm is applied
            x = self.layer_norm(x)

        out = self.feed_forward(x)
        return out
```


So ultimately the behavior is determined by the YAML configuration. In the provided YAML files we observe that the encoders and decoders are explicitly configured with "pre".

    ../joeynmt/configs/iwslt14_deen_bpe.yaml

``` yaml
encoder:
        type: "transformer"
       ...
        layer_norm: "pre"
        activation: "relu"
    decoder:
        type: "transformer"
        ...
        layer_norm: "pre"
        activation: "relu"
```
This means that for this model setup, JoeyNMT is running in pure pre-norm mode, overriding the default. These configurations make it possible for users to switch between "pre" and "post" by simply changing the layer_norm setting in a file.


DO WE NEED TO SAY WHERE LAYERNORM IS EVERYWHERE? LIKE BEFORE FF/AFTER MHATT?

# Task 2: Implementing Pre- and Post-Normalization

To implement pre- and post-normalization we need to configure two new config files where we added 

```yaml
layer_norm: "pre"
```
OR

```yaml
layer_norm: "post"
```

respectively to both the enocder and decoder layer to the default model configurations.

Both of our folders (joeynmt & mt-exercise-03) were located on Desktop, so you will need to change the paths in the config files to run them on your machine.

Due to version conflict we also downgraded Numpy module using ```pip install "numpy<2"```.

Trying to run ```train.sh``` we ran into an error that we weren't able to solve:

```yaml
 File "/Users/sud/Desktop/joeynmt/joeynmt/helpers.py", line 508, in resolve_ckpt_path
    assert load_model.is_file(), load_model
AssertionError: /Users/sud/Desktop/mt-exercise-03/models/deen_transformer_postnorm/0.ckpt
```
We tried to use the argument "--skip_test" to see if we can train the model without testing but that also didn't work.
(Out of desperation we asked ChatGPT and we tried most of it's solutions but still nothing worked.)

**Our theoretical solution :)**

From prior research (e.g., Wang et al., 2019), it is known that pre-norm configurations tend to provide greater training stability, particularly in deeper Transformer architectures. In contrast, post-norm configurations are more prone to issues such as exploding or vanishing gradients, especially when applied to deep models. 
In low-resource settings pre-norm is generally expected to perform better, particularly in the early stages of training.
Our experimental setup differed from that of Wang et al. in several key aspects: it involved significantly fewer layers(four encoder layers and one decoder layer), a much smaller dataset, and CPU-based training. Despite these differences, we would still expect pre-norm to outperform post-norm in terms of early convergence and training stability, although the final performance differences would likely remain modest due to the limited scale of the data.

In conclusion, while we were unable to complete the training :( drawing on insights from the literature and the design of this exercise, we expect that pre-norm would yield better results than post-norm in terms of validation perplexity and overall training stability in this low-resource scenario.
