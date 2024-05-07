import mysql.connector
from mysql.connector.connection_cext import CMySQLConnection
import os 
import json


path = __file__ 

confidential_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'confidential','_confidential.json'
)


def get_connection(
    confidentail_path:str = confidential_path, 
    host:str = None,  
    user:str = None, 
    password:str = None,  
    database:str = None,
    port:int = None
    )->CMySQLConnection:
    
    con:CMySQLConnection = None
    if confidentail_path:
        confidentials = json.load(open(confidentail_path))
        host, user, password, port = (
            confidentials['host'],
            confidentials['user'],
            confidentials['password'],
            confidentials['port']
        )
        if "database" in confidentials.keys():
            database = confidentials['database']
    if all([host, user, password, port]):
        print(f"connecting to {host}{port}/{database if database else ''}")
        con = mysql.connector.connect(
                host = host,
                user = user,
                password = password,
                port = port,
                database = database
        )
    else:
        raise Exception("plese provide host,user,password and port")
    
    return con