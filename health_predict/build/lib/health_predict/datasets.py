# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:52:53 2018

@author: sxx
"""
import psycopg2
import psycopg2.extras
import csv

#全局变量初始化
db_name='dsmartml'
db_user='postgres' 
db_host='60.60.60.72'
db_port='5433'
db_password='dsmart@DFC_2018'
db_uid='210100060'
os_uid='110100178'
db_dsn = "dbname=%s user=%s password=%s host=%s port=%s" % (db_name,db_user,db_password,db_host,db_port)
csv_file='dataset_133.csv'

sql_get_columns="select distinct iname,index_id from mon_indexdata_his where uid in ('%s','%s') \
and ((index_id>=2180000 and index_id<2190000 and index_id != 2180516) \
or (index_id>=3000000 and index_id != 3000300)) order by index_id" % (db_uid,os_uid)
sql_get_dataset="select * from mon_indexdata_his where uid in ('%s','%s') and \
((index_id>=2180000 and index_id<2190000 and index_id != 2180516 ) \
or (index_id>=3000000 and index_id != 3000300)) \
and (record_time >= '2018-06-01 00:00:00' and record_time < '2018-07-01 00:00:00') order by record_time,index_id asc" % (db_uid,os_uid)

db_name_1='dsmart_2018'
db_dsn_1 = "dbname=%s user=%s password=%s host=%s port=%s" % (db_name_1,db_user,db_password,db_host,db_port)
health_check_id = 176
csv_file_health = 'dataset_health_176.csv'
sql_get_columns_health="select distinct metric_id,iname from h_health_check_detail where metric_id is not null order by metric_id"
sql_get_score="SELECT * from h_health_check_deduct where health_score >1 and health_check_id=(%s) and (record_time >= '2018-06-01 00:00:00' and record_time < '2018-07-01 00:00:00') order by record_time" % (health_check_id)
sql_get_detail="SELECT * from h_health_check_detail where metric_id is not null and health_check_id=(%s) and (record_time >= '2018-06-01 00:00:00' and record_time < '2018-07-01 00:00:00') order by record_time" % (health_check_id)
sql_get_itemscore="select * from h_health_check_item_score where health_check_id=(%s) and (record_time >= '2018-06-01 00:00:00' and record_time < '2018-07-01 00:00:00') order by record_time" % (health_check_id)


'''获取健康模型数据集的所有列名
参数：
    sql:查询所有列名的sql语句
返回：
    list类型，包含所有数据列
'''
def get_columns(sql,header=[]):
    with psycopg2.connect(db_dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor) as tuple_cur:
            tuple_cur.execute(sql)
            rows=tuple_cur.fetchall()
            
            for row in rows:
                if row.iname is None:
                    column = "%s" % row.index_id
                else:
                    column = "%s(%s)" % (row.index_id,row.iname)
                header.append(column)


def get_columns_health(sql,header):
    with psycopg2.connect(db_dsn_1) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as dict_cur:
            dict_cur.execute(sql)
            rows=dict_cur.fetchall()
            
            for row in rows:
                if row['iname'] is None:
                    column = "%s" % row['metric_id']
                else:
                    column = "%s(%s)" % (row['metric_id'],row['iname'])
                header.append(column)

'''获取健康模型数据集并存入csv文件
参数：
    columns:需要查询的数据列
    csv_file:数据集写入的文件名
    sql:查询数据集的sql语句
'''
def get_total_datasets(columns,csv_file,sql):
    with psycopg2.connect(db_dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor) as tuple_curs:
            tuple_curs.execute(sql)
            rows = tuple_curs.fetchall()

            dataset = {}
            for row in rows: 
                time = row.record_time.strftime("%Y%m%d%H%M%S")
                mainkey = "%s-%s-%s" % (time,db_uid,os_uid)
                if mainkey not in dataset:
                    dataset[mainkey]={}
                if row.iname is None:
                    column = "%s" % row.index_id
                else:
                    column = "%s(%s)" % (row.index_id,row.iname)
                dataset[mainkey][column] = row.value
                dataset[mainkey]['record_time']=row.record_time

            with open(csv_file,'w',newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns,delimiter=',')
                writer.writeheader()
                lines = 0                
                for key in dataset:
                    row = dataset[key]
                    writer.writerow(row)
                    lines += 1


def get_health_datasets(columns,csv_file,sql_get_score,sql_get_detail,sql_get_itemscore):
    health_dataset={}
    with psycopg2.connect(db_dsn_1) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as dict_curs:
            dict_curs.execute(sql_get_score)
            score_set=dict_curs.fetchall()            
            dict_curs.execute(sql_get_detail)
            detail_set=dict_curs.fetchall()            
            dict_curs.execute(sql_get_itemscore)
            itemscore_set=dict_curs.fetchall()

        for row in score_set:
            main_key = '%s_%d' % (row['record_time'].strftime("%Y%m%d%H%M%S"), row['health_check_id'])           
            if main_key not in health_dataset:
                health_dataset[main_key] = {}
            health_dataset[main_key]['record_time'] = row['record_time']
            health_dataset[main_key]['health_score'] = row['health_score']          
        for row in itemscore_set:
            main_key = '%s_%d' % (row['record_time'].strftime("%Y%m%d%H%M%S"), row['health_check_id'])
            if main_key in health_dataset:
                health_dataset[main_key][row['model_item_id']] = row['score']
            #print ("itemscore----",row['model_item_id'],"-->",row['score'])
        #print ("health_dataset",health_dataset)
        for row in detail_set:
            if row['iname'] is None:
                column = "%s" % row['metric_id']
            else:
                column = "%s(%s)" % (row['metric_id'],row['iname'])
            main_key = '%s_%d' % (row['record_time'].strftime("%Y%m%d%H%M%S"), row['health_check_id'])
            if main_key in health_dataset:
                health_dataset[main_key][column] = row['metric_value']
        
        with open(csv_file,'w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns,delimiter=',')
            writer.writeheader()
            lines = 0                
            for key in health_dataset:
                row = health_dataset[key]
                writer.writerow(row)
                lines += 1
                
if __name__ == '__main__':
    #header = get_columns(sql_get_columns)
    header=['record_time']
    get_columns(sql_get_columns,header)
    #print ("len:",len(header),"columns",header)
    get_total_datasets(header,csv_file,sql_get_dataset)
    
    header_health=['record_time','health_score',67,68,69,70,71,72,73]
    get_columns_health(sql_get_columns_health,header_health)
    print ("len:",len(header_health),"columns",header_health)
    get_health_datasets(header_health,csv_file_health,sql_get_score,sql_get_detail,sql_get_itemscore)
    