library('tseries')
library('TSA')
library('stats')

# Récupérer les données par jour :
dossier = 'C:/DocDataM2/S3UE3_Serie_chrono/projet1'
fichier = 'export_fr_par_heure_1000_derniere.csv'

library("readr")
data = read.csv(paste(dossier,fichier, sep='/'))
head(data)

# transformer le format date
Date = as.POSIXct(strptime(data$publishedAt, "%Y-%m-%dT%H"))
Traffic <- as.numeric(data$title)
data<-data.frame(Date,Traffic)
head(data)
data$Traffic

library('forecast')
#plot(boxcox(data$Traffic,0.1), type='l')

plot(data, type='l')
Ldata_T = log(data$Traffic)
plot(Ldata_T, type='l')

length(Ldata_T)

Ldata_T = Ldata_T[1:300]

"""
jour = 4
plot(Ldata_T[(24*(jour-1)):(jour*24)], type='l')
"""
data_ts = ts(data = Ldata_T, start = 1, frequency = 24)
plot(data_ts)

library("tseries")
adf.test(data_ts)   # H0 Non Stationnaire   : REJETE
kpss.test(data_ts)  # H0 Stationnaire       : ACCEPTE



# Un coup d'oeil mois par mois par mois
library('forecast')
seasonplot(data_ts)

#dev.off()


# Modélisation additive sur la série complète
Decomp = decompose(data_ts, type="additive")
plot(Decomp) # La fonction 'decompose' propose un affichage condensé des composantes


# Voyons ça de plus près...
plot(Decomp$trend, type="l", col="blue", ylab="Tendance")
plot(Decomp$seasonal, type="l", col="red", ylab="Périodicité")
plot(Decomp$figure, type="l", col="magenta", ylab="Motif périodique")
plot(Decomp$random, type="l", col="forestgreen", ylab="Fluctuation")

Decomp$random

partie_rand = Decomp$random[13:288]

adf.test(partie_rand)
kpss.test(partie_rand)



# Série tronquée pour reprédiction
n = length(LData)
f = 12
LDataT = ts(LData[1:(n-f)], frequency=12, start=1949) # Il faut remettre la version tronquée au format ts pour le 'decompose'
DernPer = LData[(n-f+1):n]
DecompT = decompose(LDataT, type="additive")





"""
# Régression linéaire sur la tendance
Tps = 1:(n-f)
RegLinTrend = lm(DecompT$trend ~ Tps)
summary(RegLinTrend)
estA0 = RegLinTrend$coefficients[1]
estA1 = RegLinTrend$coefficients[2]
plot(Tps, DecompT$trend, type="l", ylab="Tendance")
lines(Tps, estA0 + estA1*Tps, col="red", lty=2)

# Reprédiction de la nouvelle période
NTps = (n-f+1):n
PredT = estA0 + estA1*NTps
PredS = DecompT$figure
Pred = PredT + PredS

# Superposition série/prédiction dernière période
plot(LData, type="l")
lines(ts(Pred, frequency=12, start=1960), col="red", lwd=2)

# Un critère de qualité de la prédiction (pour comparaison de modèles prédictifs)
MSE = sum((DernPer - Pred)^2)/f

# Prédiction de deux nouvelles périodes avec le modèle additif
Tps = 1:n
Decomp = decompose(LData, type="additive")
RegLinTrend = lm(Decomp$trend ~ Tps)
estA0 = RegLinTrend$coefficients[1]
estA1 = RegLinTrend$coefficients[2]
NTps = (n+1):(n+2*f)
PredT = estA0 + estA1*NTps
PredS = Decomp$figure
Pred = PredT + PredS
plot(ts(c(LData, Pred), frequency=12, start=1949), type="l")
lines(ts(Pred, frequency=12, start=1961), col="red", lwd=2)

# Et... n'oublions pas que la série initiale avait été passée au log
plot(ts(c(Data, exp(Pred)), frequency=12, start=1949), type="l")
lines(ts(exp(Pred), frequency=12, start=1961), col="red", lwd=2)

"""