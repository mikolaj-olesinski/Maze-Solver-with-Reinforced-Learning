import pyodbc
import os
from dotenv import load_dotenv

load_dotenv()

server = os.getenv('SQL_SERVER')
database = os.getenv('SQL_DATABASE')
username = os.getenv('SQL_USERNAME')
password = os.getenv('SQL_PASSWORD')
driver = '/opt/homebrew/Cellar/msodbcsql17/17.10.6.1/lib/libmsodbcsql.17.dylib'

conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};Authentication=ActiveDirectoryPassword'
conn = pyodbc.connect(conn_str)
c = conn.cursor()

def create_database():
    c.execute('''
        CREATE TABLE files (
            id INTEGER PRIMARY KEY IDENTITY,
            name NVARCHAR(MAX) NOT NULL,
            grid NVARCHAR(MAX) NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            start_x INTEGER NOT NULL,
            start_y INTEGER NOT NULL,
            end_x INTEGER NOT NULL,
            end_y INTEGER NOT NULL
        );

        CREATE TABLE agents (
            id INTEGER PRIMARY KEY IDENTITY,
            file_id INTEGER NOT NULL,
            file_name NVARCHAR(MAX) NOT NULL,
            best_score INTEGER NOT NULL,
            mean_score REAL NOT NULL,
            n_games INTEGER NOT NULL,
            epsilon REAL NOT NULL,
            FOREIGN KEY (file_id) REFERENCES files (id)
        );
    ''')

def add_file(name, grid, width, height, start_x, start_y, end_x, end_y):
    c.execute('''
        INSERT INTO files (name, grid, width, height, start_x, start_y, end_x, end_y)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, grid, width, height, start_x, start_y, end_x, end_y))
    conn.commit()

def grid_str_to_list(grid_str):
    grid = []
    for line in grid_str.strip().split("\n"):
        if line.strip():
            grid.append(list(map(int, line.strip().split())))
    return grid

def get_grid_by_id(file_id):
    c.execute('''
        SELECT grid FROM files WHERE id = ?
    ''', (file_id,))

    return grid_str_to_list(c.fetchone()[0])

def add_agent(file_id, file_name, best_score, mean_score, n_games, epsilon):
    c.execute('''
        INSERT INTO agents (file_id, file_name, best_score, mean_score, n_games, epsilon)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (file_id, file_name, best_score, mean_score, n_games, epsilon))
    conn.commit()

def list_tables():
    c.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE';")
    tables = c.fetchall()
    for table in tables:
        print(table[0])

if __name__ == "__main__":
    list_tables()
    conn.close()





