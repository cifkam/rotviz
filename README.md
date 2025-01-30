# Video Annotation Tool

## Installation


```bash
pip install git+ssh://git@github.com/cifkam/vizrot.git
```

### You can  use virtual environment

```bash
pdm venv create -v --with-pip 3.11
# or
conda create -n venv python=3.11 'libstdcxx-ng>=13' -c conda-forge
```

## Usage

```bash
python -m rotviz --mesh <path_to_mesh>
```

```
y - change rotation multiplier between 1, 5, 10 degrees

z - increase depth by depth_multiplier
x - decrease depth by depth_multiplier

a - rotate by  1*rotation_multiplier degrees along camera y
d - rotate by -1*rotation_multiplier degrees along camera y

w - rotate by  1*rotation_multiplier degrees along camera x
s - rotate by -1*rotation_multiplier degrees along camera x

q - rotate by  1*rotation_multiplier degrees along camera z
e - rotate by -1*rotation_multiplier degrees along camera z

f - reset rotation

esc - exit
```
