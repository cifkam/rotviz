# Rotation Visualization Tool

## Installation


```bash
pip install git+ssh://git@github.com/cifkam/rotviz.git
```

### You can  use virtual environment

```bash
conda create -n venv python=3.11 'libstdcxx-ng>=13' -c conda-forge
```

## Usage

```bash
python -m rotviz --data <path_to_json_data>
```

Rotate the mesh with left mouse button or with keys:
```
y - change rotation multiplier between 1, 2.5, 5, 10 degrees
u - change depth multiplier between 1, 5, 10

z - increase depth by depth_multiplier
x - decrease depth by depth_multiplier

a - rotate by  1*rotation_multiplier degrees along camera y
d - rotate by -1*rotation_multiplier degrees along camera y

w - rotate by  1*rotation_multiplier degrees along camera x
s - rotate by -1*rotation_multiplier degrees along camera x

q - rotate by  1*rotation_multiplier degrees along camera z
e - rotate by -1*rotation_multiplier degrees along camera z

f - reset rotation

1 - set camera forward axis to +x, i.e. [ 1,  0,  0]
4 - set camera forward axis to -x, i.e. [-1,  0,  0]

2 - set camera forward axis to +y, i.e. [ 0,  1,  0]
5 - set camera forward axis to -y, i.e. [ 0, -1,  0]

3 - set camera forward axis to +z, i.e. [ 0,  0,  1]
6 - set camera forward axis to -z, i.e. [ 0,  0, -1]

esc - exit
```
