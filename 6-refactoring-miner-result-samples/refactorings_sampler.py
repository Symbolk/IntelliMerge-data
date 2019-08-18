import os
import pandas as pd
from sqlalchemy import create_engine
import sys
import re
import numpy as np

db_connection = ""


def get_db_connection():
    username = 'root'
    password = 'passwd'
    database_name = 'refactoring_analysis'
    server = '127.0.0.1'

    with open("./database.properties", 'r') as db_file:
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
    df = pd.read_sql(query, db_connection)
    return df.iloc[0]['id']


def get_refs_by_project_id(project_id):
    query = "SELECT * FROM refactoring WHERE project_id={}".format(project_id)
    df = pd.read_sql(query, db_connection)
    return df


def get_ref_details_by_id(project_id, ref_id):
    query = "SELECT * FROM refactoring_region WHERE project_id={} AND refactoring_id={}".format(project_id, ref_id)
    df = pd.read_sql(query, db_connection)
    return df


def print_to_csv(path, entry):
    with open(path, 'a') as open_a:
        open_a.write('-' * 20 + '\n' + entry + '\n')


if __name__ == '__main__':
    # connect to the database
    db_connection = get_db_connection()
    project_names = []
    project_names.append("junit")
    project_names.append("error-prone")
    project_names.append("realm-java")
    project_names.append("gradle")
    project_names.append("javaparser")
    project_names.append("storm")
    project_names.append("deeplearning4j")
    project_names.append("antlr4")
    project_names.append("elasticsearch")
    project_names.append("cassandra")

    for project_name in project_names:
        project_id = get_project_id(project_name)

        # get the data and the id range
        refs = get_refs_by_project_id(project_id)
        start = refs.iloc[0]['id']
        refs_num = len(refs.index)
        # sample 10% for each project
        sample_refs_num = 0.1 * refs_num

        print('Start with: ' + project_name)

        path = project_name + '_sample_refactorings.csv'
        if not os.path.isfile(path):
            with open(path, "w") as open_w:
                # header
                open_w.write(project_name + ': ' + str(round(sample_refs_num)) +
                             '/' + str(refs_num) + '\n')

        # randomly sample 10(%), save record in csv file with the project_name with name
        np.random.seed(3)
        for i in range(round(sample_refs_num)):
            # print(np.random.random())
            index = round(np.random.random() * refs_num)
            ref = refs.iloc[index]
            ref_id = ref['id']
            ref_details = get_ref_details_by_id(project_id, ref_id)

            # all info need to validate the ref
            entry = ''
            line = []
            line.append(ref['commit_hash'])
            line.append(ref['refactoring_type'])
            line.append(ref['refactoring_detail'])
            entry = ';'.join(line)
            for _, row in ref_details.iterrows():
                line = []
                line.append(row['type'])
                line.append(row['path'])
                line.append(str(row['start_line']))
                line.append(str(row['length']))
                entry += '\n' + ';'.join(line)
            print_to_csv(path, entry)
        print('Done with: ' + project_name)
