# ASTRA: Agenda-based generation of Semantic TRAjectories

Human mobility data is valuable for many applications, but poses a significant privacy risk for individuals. A person's daily movements are closely linked to their socio-demographic characteristics and their points of interest (POI), which can reveal sensitive information about them, such as their religion or educational background. 
Researching and mitigating these risks requires realistic, semantically rich data, which is often unavailable or lacks semantic features. 
We introduce ASTRA, an agenda-based approach for generating synthetic mobility data with semantic attributes. 
Using real travel surveys and census data, ASTRA simulates artificial agents with socio-demographic features that follow daily activity agendas. It maps activities to semantically similar POIs, and projects them onto real-world locations within a user-defined geographical region. 
To model movement, ASTRA extends the exploration and preferential return (EPR) model with spatio-temporal and semantic constraints as imposed by an agenda. 
Our evaluation against real check-in data and the EPR baseline model shows that ASTRA can generate realistic mobility data at scale, preserving important characteristics of human movement.


### Build
#### Set up database
To set up a PostgreSQL database with PostGIS, first, update the Dockerfile with your credentials. Then, create and 
start the docker container by running
```bash
sudo docker build . -t postgis -f Dockerfile
sudo docker run --name astra -d -p 5438:5438 postgis
```
The database is running on localhost.

Test the database connection and list all available databases by running
```bash
sudo psql -l -p 5438 -h localhost -U user
```

If you want to explore the database, login to get a psql console by running
```bash
sudo psql -d astra -U user -p 5438 -h localhost
```

#### Get required resources
1) Download Population count at [WorldPop](https://hub.worldpop.org) for the desired country and place it in the folder specified in the [config file](configs/config.toml).
2) Download population distribution per age and sex groups at [WorldPop](https://hub.worldpop.org) for the desired region and place it in the folder specified in the [config file](configs/config.toml).
3) Download [MTUS data set](https://www.mtusdata.org/mtus/) for the desired region and time range for the desired country and place it at the path specified in the [config file](configs/config.toml). Required columns to select can be found in ```bash load_df() ```.
in [survey_loading.py](src/survey_processing/loading/survey_loading.py).

#### Configuration
Update parameters, if required, in [config.toml](configs/config.toml) and [run.toml](configs/run.toml).


## Authors and acknowledgment
*ASTRA* is developed by University of Leipzig & ScaDS.AI Dresden/Leipzig, Germany, funded by the Federal Ministry of 
Education and Research of Germany and by Sächsische Staatsministerium für Wissenschaft, Kultur und Tourismus in the 
program Center of Excellence for AI-research ”Center for Scalable Data Analytics and Artificial Intelligence 
Dresden/Leipzig”, project identification number: ScaDS.AI.

This work was also partially funded by Universities Australia and the German Academic Exchange Service (DAAD) grant 57701258. 

This study further uses the Multinational Time Use Study, Centre for Time Use Research, University College London 2019 ([http://www.timeuse.org/mtus/reference.html](http://www.timeuse.org/mtus/reference.html)).

## License
*ASTRA* is licensed under [Apache License, Version 2.0](LICENSE)
