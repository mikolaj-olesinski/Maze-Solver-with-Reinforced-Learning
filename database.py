import sqlite3

conn = sqlite3.connect('maze_data.db')
c = conn.cursor()


def create_database():
    c.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            name TEXT,
            grid TEXT,
            width INTEGER,
            height INTEGER,
            start_x INTEGER,
            start_y INTEGER,
            end_x INTEGER,
            end_y INTEGER
        )
    ''')
            
    c.execute('''
        CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY,
            file_id INTEGER REFERENCES files(id),
            file_name TEXT,
            best_score INTEGER,
            mean_score REAL,
            n_games INTEGER,
            epsilon REAL
        )
    ''')
        
def add_file(name, grid, width, height, start_x, start_y, end_x, end_y):
    c.execute('''
        INSERT INTO files (name, grid, width, height, start_x, start_y, end_x, end_y)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, grid, width, height, start_x, start_y, end_x, end_y))
    conn.commit()
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

if __name__ == "__main__":
    create_database()
    conn.close()