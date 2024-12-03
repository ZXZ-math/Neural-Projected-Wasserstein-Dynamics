Paper link: https://arxiv.org/pdf/2402.16821

## Prerequisites

Ensure you have **Poetry** installed. If Poetry is not installed on your system, 
```
curl -sSL https://install.python-poetry.org | python3 -
```
Navigate to the project folder, 
`
poetry install 
`
 And then 
`
poetry shell
`. 

## Run code 
Run `python xxx.py` replace xxx.py with `linear_transport.py` (for transport equation), `fokker_planck.py` (for fokker planck equation), `porous_medium.py` (for poroud medium equation), `KS.py` (for keller-segel equation). 

When running `porous_medium.py`, you will need to also modify `compute_G.py`: comment out the Gaussian part and uncomment the porous medium part. When running other PDEs, make sure to comment out the porous medium part and uncomment the Gaussian part. 