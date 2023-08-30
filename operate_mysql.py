import pymysql
import logging
import configparser

class do_mysql(object):
    def __init__(self):
        con = configparser.ConfigParser()
        con.read('./config.ini', encoding='utf-8')
        section = 'SQL'
        host = con.get(section, 'host')
        port = con.getint(section, 'port')
        user = con.get(section, 'user')
        password = con.get(section, 'password')
        database = con.get(section, 'database')
        charset = con.get(section, 'charset')
        conn = pymysql.connect(host=host,
                               port=port,
                               user=user,
                               password=password,
                               database=database,
                               charset=charset
                               )
        self.conn = conn

    def insert_dict(self, table_name, **kwargs):
        conn = self.conn
        cursor = conn.cursor()
        keys = ', '.join('`{}`'.format(key) for key in kwargs.keys())
        values = ', '.join('%({})s'.format(key) for key in kwargs.keys())
        sql = "insert into {}({}) VALUES({})".format(table_name, keys, values)
        cursor.execute(sql, kwargs)
        conn.commit()


    def update(self, table_name, conditions=None, **kwargs):
        conn = self.conn
        cursor = conn.cursor()
        cols = ", ".join('`{}`=%({})s'.format(k, k) for k in kwargs.keys())
        if not conditions:
            condition = 'where id=%(id)s'
        else:
            convalue = ' and'.join('`{}`=%({})s'.format(item[0], item[0]) for item in conditions.items())
            condition = 'where {}'.format(convalue)
        sql = """update %s set %s %s""" % (table_name, cols, condition)
        cursor.execute(sql, kwargs)
        conn.commit()


    def select(self, table_name, conditions=None):
        conn = self.conn
        cursor = conn.cursor()
        conn.cursor()
        if conditions:
            keys = " and ".join('`{}`=%({})s'.format(k, k) for k in conditions.keys())
            condition = 'where {}'.format(keys)
            sql = "select * from %s %s" %(table_name, condition)
            cursor.execute(sql, conditions)
        else:
            sql = "select * from %s " % (table_name)
            cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
        return result