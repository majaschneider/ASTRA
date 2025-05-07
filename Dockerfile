FROM postgis/postgis:latest

ENV POSTGRES_USER=user
ENV POSTGRES_PASSWORD=password

EXPOSE 5438

ENV POSTGRES_DB=astra
ENV PGPORT=5438

WORKDIR /app

COPY init.sql /docker-entrypoint-initdb.d/

# Start the PostgreSQL server
CMD ["postgres"]
