import os
import pandas as pd
from sqlalchemy import create_engine
import sys
import re
import numpy as np


def get_db_connection():
    username = 'root'
    password = 'passwd'
    database_name = 'refactoring_analysis'
    server = '127.0.0.1'

    with open("../database.properties", 'r') as db_file:
        for line in db_file:
            line = line.strip()
            username_search = re.search('^development.username=(.*)$', line, re.IGNORECASE)
            password_search = re.search('^development.password=(.*)$', line, re.IGNORECASE)
            url_search = re.search('^development.url=jdbc:mysql://(.*)/(.*)$', line, re.IGNORECASE)

            if username_search:
                username = username_search.group(1)
            if password_search:
                password = password_search.group(1)
            if url_search:
                server = url_search.group(1)
                database_name = url_search.group(2)

    return create_engine('mysql+pymysql://{}:{}@{}/{}'.format(username, password, server, database_name))

def get_project_id(project_name):
    query = "SELECT * FROM project WHERE name='{}'".format(project_name)
    df = pd.read_sql(query, get_db_connection())
    return df.iloc[0]['id']

def get_refs_by_project_id(project_id):
    query = "SELECT * FROM refactoring WHERE project_id={}".format(project_id)
    df = pd.read_sql(query, get_db_connection())
    return df

def print_to_csv(path, line):
    if not os.path.isfile(path):
        with open(path, "w") as open_w:
            # header
            open_w.write(
                "id;merge_commit;parent1;parent2;merge_base;ref_type;ref_detail;old_path;old_start_line;new_path;new_start_line")
    with open(path, 'a') as open_a:
        open_a.write('\n' + ';'.join(line))

if __name__ == '__main__':
    project_name = 'javaparser'
    project_id=get_project_id(project_name)

    # get the data and the id range
    refs = get_refs_by_project_id(project_id)
    start = refs.iloc[0]['id']
    refs_num = len(refs.index)

    # randomly sample 10(%), save record in csv file with the project_name with name
    np.random.seed(3)
    path='results/temp.csv'
    for i in range(10):
        # print(np.random.random())
        index = round(np.random.random() * refs_num)
        print_to_csv(path, refs.iloc[index].T)