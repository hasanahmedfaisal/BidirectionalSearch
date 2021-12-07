## "Meet in the middle" Bidirectional Search integrated into Pacman Domain

This repository implements bidirectional search as described in the paper **"Bidirectional Search That Is Guaranteed to Meet in the Middle"** by Robert C. Holte, Ariel Felner, Guni Sharon and Nathan R. Sturtevant in the Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16).

The code for the search can be found in [search.py](https://github.com/deepakshi-mittal/bds_search/blob/BDSFinalProject/search/search.py)

In order to see pacman in action, the following are commands for six different mazes:
```
python pacman.py -l tinyMaze -p SearchAgent -a fn=bdMM0

python pacman.py -l smallMaze -p SearchAgent -a fn=bdMM0

python pacman.py -l mediumMaze -p SearchAgent -a fn=bdMM0

python pacman.py -l bigMaze -p SearchAgent -a fn=bdMM0

python pacman.py -l openMaze -p SearchAgent -a fn=bdMM0

python pacman.py -l contoursMaze -p SearchAgent -a fn=bdMM0

```
You can speed up Pacman by adding ```--frameTime 0``` and to change search strategy change ```fn``` to one of:
```
bfs ===> Breadth First Search
dfs ===> depth First Search
astar ===> Astar Search
ucs ===> Uniform Cost Search
bdMM0 ===> Bidirectional Brute Force
bdMM ===> Bidirectional Meet in the Middle
```
The below command executes the Meet in the Middle Bidirectional Algorithm with food heuristic which was implemented:
```
python pacman.py -l contoursMaze -p SearchAgent -a fn=bdMM,heuristic=foodHeuristic
```

Implemented using Python 3.6
