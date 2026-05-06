import psycopg


def get_connection(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn)
