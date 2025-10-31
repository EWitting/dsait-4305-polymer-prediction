REM Text-based
uv run main.py model=gcn preprocessing=text_based model.in_channels=43
uv run main.py model=gat preprocessing=text_based model.in_channel=43
uv run main.py model=hrg preprocessing=text_based model.in_channels=43


REM Oligomer size 2
uv run main.py model=gcn preprocessing.oligomer_len=2
uv run main.py model=gat preprocessing.oligomer_len=2 
uv run main.py model=hrg preprocessing.oligomer_len=2 


REM Oligomer size 3
uv run main.py model=gcn preprocessing.oligomer_len=3
uv run main.py model=gat preprocessing.oligomer_len=3 
uv run main.py model=hrg preprocessing.oligomer_len=3 

REM Oligomer size 4
uv run main.py model=gcn preprocessing.oligomer_len=4
uv run main.py model=gat preprocessing.oligomer_len=4 
uv run main.py model=hrg preprocessing.oligomer_len=4 