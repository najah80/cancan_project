#Importer toutes les librairie nécessaires
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

from math import *
from collections import Counter
import datetime

spark = SparkSession.builder.getOrCreate()

# récupérer les données 3G  pour deux semaines (du 16/03/2019 au 31/03/2019)
day = "2019/03/25"
dt_day = datetime.datetime.strptime(day,"%Y/%m/%d")

iu = spark.read.parquet("/Data/P17100/SECURE/RAW/TXT/IU/2019/03/*/").filter(F.col('imsi').startswith('20801')).where(~F.col('Tac').isNull()).withColumnRenamed('IMSI', 'imsi').withColumnRenamed('Tac', 'TAC').withColumn('ts',F.unix_timestamp(F.col("End_Time"), "dd/MM/yyyy HH:mm:ss").cast('timestamp'))

iu = iu.filter(iu.ts.between(dt_day,dt_day + datetime.timedelta(days=8))).filter(~iu.First_Event.startswith("LUREQ") & ~iu.First_Event.startswith("GMM_RA_UPD"))

iu_event = {"CALL" : F.col('Service_Type') == 'Voice', "SMS" : F.col('Service_Type') == 'SMS', "DATA" : F.col('Service_Type') == 'Data PS'}

#Créer une colonne service (data, sms, call)
iu = iu.withColumn("service",F.when(iu_event['SMS'],'SMS').when(iu_event['CALL'],'CALL').when(iu_event['DATA'],'DATA')).select('imsi','TAC','service','ts', 'SAI_LAC', 'SAC_Value')

# récupérer les fichiers topo3G 
topo = spark.read.csv("/Data/P17100/SECURE/SHARE/TOPO/ALL_CELLS_NIDT", header='True', sep=';').filter("'{}' between min_dt and max_dt".format(day))

topo3G = topo.filter("TECHNO='3G'").select(topo.CI.alias("SAC_Value"),topo.LAC.alias("SAI_LAC"),F.struct(topo.COORD_X.cast('float'), topo.COORD_Y.cast('float')).alias('pos'))

# jointure avec les fichiers topo
iu_joined = iu.join(topo3G, on=["SAI_LAC","SAC_Value"], how='left')

# les données 3G après jointure avec topo
df_3G = iu_joined.select('imsi', 'TAC', 'ts', 'service', 'pos')

# prendre les valeurs uniques de imsi avec tous les imei uniques et non null & rajouter une colonne "size" pour compter le nombre d'imei pour chaque imsi
imsi_tac = df_3G.select('imsi', 'TAC', 'ts', 'service', 'pos').groupBy("imsi").agg(F.collect_set('TAC').alias("TAC"), F.collect_list('service').alias("service_list"), F.collect_list(F.col('pos')).alias('positions'), F.collect_set("ts").alias('events_list')).withColumn('nb_events', F.size(F.col('events_list'))).withColumn("size", F.size(F.col('TAC')))


# les imsi avec un seul code imei
imsi_1tac = imsi_tac.filter(F.col("size") == '1')
imsi_1tac = imsi_1tac.withColumn("TAC", imsi_1tac["TAC"].getItem(0).cast("string"))

# récupérer le fichier TAC labélisé (pour chaque code TAC (MACHINE ou Non))
df_TAC = spark.read.csv("/Data/P17100/SECURE/SHARE/TAC_DEVICE_201909.csv", header='True', sep='\t')

# jointure des imsi (avec un seul imei) avec le fichier TAC 
df_3G_joined = imsi_1tac.join(df_TAC, ['TAC'], how='left')


def occurence(liste):
    """
    une udf pyspark pour
    calculer le max d'occurence des apns
    retourne le max d'occurence et l'apn

    """
    if len(liste) > 0:
    
         liste_occ = [(liste.count(elem), elem) for elem in set(liste)]
         max_occ = max(liste_occ, key=lambda l:l[0])
    
         return  max_occ

schema = StructType((
    StructField("count", IntegerType(), True),
    StructField("char", StringType(), True)
))
occ_udf = F.udf(occurence, schema)


def day_night(user_events):
    """
    Calcule le nombre d'évènements par jour et par nuit 
    retourne le nombre par jour(entre 6h et 21h) et le nombre d'évènement par nuit en prenant une partie de la nuit précédente N-1 (de         minuit à 6h) 
    et une partie de la nuit N ( de 21h à minuit)

    """
    sorted_list = sorted(user_events)
    days = (sorted_list[-1] - sorted_list[0]).days
    if (days > 3): # au moins 4 jours sur 7 d'observation

        day_ts = range(7, 22)
        day = [x for x in sorted_list if x.hour in day_ts]
        d = len(day)
        n = len(sorted_list) - d
     
        return (d,n)
schema1 = StructType((
StructField("nb_event_day", IntegerType(), True),
StructField("nb_event_night", IntegerType(), True)
))
dn_udf = F.udf(day_night, schema1)


def inter_event_time(user_events):
    """
    Calcule l'inter_event_time 
    retourne la moyenne de tous les inter_time
    Calcule l'entropie des inter_events time  
    retourne l'entropie
  
    """     

    sorted_list = sorted(user_events)
    days = (sorted_list[-1] - sorted_list[0]).days

    if (days > 3): # au moins 4 jours sur 7 d'observation
        diff = [(t - s).total_seconds() for s, t in zip(sorted_list, sorted_list[1:]) if (t-s).total_seconds() > 1]

        l1 = len(diff)

        if l1 > 3:# on prend les imsi avec au moins 4 events par jours

            X =  [(elem, diff.count(elem)) for elem in set(diff)]
             
            x0 = X[0][1]
        
            entropy = (x0/l1) * log(x0/l1)
        
            for x in X[1:l1+1]:
            
                 entropy+= (x[1]/l1 ) * log(x[1]/l1)
            
            entropy = -round((entropy / log(l1)), 2)
        
            diff_avg = sum(diff) // len(sorted_list)

            return (int(diff_avg), entropy)   

schema2 = StructType((
StructField("ie_avg", IntegerType(), True),
StructField("entropy", FloatType(), True)
))
ie_entropy_udf = F.udf(inter_event_time, schema2)


def radius_of_gyration(positions):
    """
    Returns the radius of gyration, the *equivalent distance* of the mass from
    the center of gravity, for all visited places. [GON2008]_.
    
    """ 
    positions = [tuple(l) for l in positions]
    d = Counter(p for p in positions
                if p is not None)

    sum_weights = sum(d.values())
    positions = list(d.keys())  # Unique positions

    if len(positions) == 0:
        return None

    bary = [0, 0]
    for pos, t in d.items():
        bary[0] += pos[0] * t
        bary[1] += pos[1] * t

    bary[0] /= sum_weights
    bary[1] /= sum_weights

    r = 0.
    for pos, t in d.items():

        arc = ((pos[0]-bary[0])**2 + (pos[1]-bary[1])**2)**0.5
        r += float(t) / sum_weights * (arc ** 2)
        
    return int(sqrt(r))
rg_udf = F.udf(radius_of_gyration, IntegerType())


def entropy_positions(positions):

    c = Counter(p for p in positions if p is not None)
    lenc = len(c.values())
    if lenc == 0:
        return None
    if lenc == 1:
        return 0
    sumc = sum(c.values())
    probas = [p/sumc for p in c.values()]
    e = 0
    for pr in probas:
        e -= pr * log(pr,2)
    e = round(e/log(sumc,2),2)
    return e
pos_entropy_udf = F.udf(entropy_positions, FloatType())


def count_services(services):
    c = Counter(services)
    return (c.get('CALL',0),c.get('SMS',0),c.get('DATA',0))

schema_services = StructType((
StructField("CALL", IntegerType(), True),
StructField("SMS", IntegerType(), True),
StructField("DATA", IntegerType(), True)
))
services_udf = F.udf(count_services, schema_services)

#Calculer les métriques
new_df_3G = df_3G_joined.select('imsi', 'TAC', 'nb_events', 'DEVICE_TYPE',  
    dn_udf(F.col('events_list')).alias('day_night'), 
    ie_entropy_udf(F.col('events_list')).alias('ie_entropy'),
    pos_entropy_udf(F.col('positions')).alias('pos_entropy'),
    rg_udf(F.col('positions')).alias('rg'),
    services_udf("service_list").alias('service'))

# sauvegarder les résultats
new_df_3G = new_df_3G.drop("imsi").repartition(1).withColumn("user_id",F.monotonically_increasing_id())
new_df_3G.write.parquet("/Data/P17100/SECURE/SHARE/MOBITIC/SAMPLE10K/M2M/results/new_df_3G_noLU_2s", mode='overwrite')




