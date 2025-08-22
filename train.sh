# hidden size experiments
uv run src/run.py --hidden 32 --experiment-name hidden_32
uv run src/run.py --hidden 64 --experiment-name hidden_64
uv run src/run.py --hidden 128 --experiment-name hidden_128
uv run src/run.py --hidden 256 --experiment-name hidden_256
uv run src/run.py --hidden 512 --experiment-name hidden_512


# number of layers experiments
uv run src/run.py --num-layers 1 --experiment-name layer_1
uv run src/run.py --num-layers 2 --experiment-name layer_2
uv run src/run.py --num-layers 3 --experiment-name layer_3
uv run src/run.py --num-layers 4 --experiment-name layer_4
uv run src/run.py --num-layers 5 --experiment-name layer_5

