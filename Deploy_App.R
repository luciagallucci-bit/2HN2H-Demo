renv::init()               
renv::snapshot()    
rsconnect::setAccountInfo(name='justgimmi', token='F196F3A6BAF647BBE2F448FB80876CA3',
                          secret='j/5+cKnkQ79y4xYPupxs2AwSW3pOcTCveGntbCXx')
rsconnect::deployApp(
   appDir = getwd(),
   appFiles = c(
     "app.R", "renv.lock",
     "combined_YAGO3_10_RotatE.parquet",
     "train_YAGO3_10.txt"
   )
 )
# 
# rsconnect::showLogs(getwd())