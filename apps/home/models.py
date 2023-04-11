from django.db import models

# Creating our models here.

class Secteur(models.Model):
    id=models.BigAutoField(primary_key=True)
    nom = models.CharField(max_length=100, unique=True)

class Societe(models.Model):
    id=models.BigAutoField(primary_key=True)
    label = models.CharField(max_length=20, unique=True) 
    nom = models.CharField(max_length=100) 
    secteur=models.ForeignKey(Secteur, on_delete=models.CASCADE)

class Historical_data(models.Model):
    id=models.BigAutoField(primary_key=True)
    Date = models.DateField()
    Open = models.FloatField(default=0)
    High = models.FloatField(default=0)
    Low = models.FloatField(default=0)
    Close = models.FloatField(default=0)
    Volume = models.FloatField(default=0)
    Rendement=models.FloatField(default=0)
    Rendement_moyen=models.FloatField(default=0)
    societe=models.ForeignKey(Societe, on_delete=models.CASCADE)

class RealTime_data(models.Model):
    id=models.BigAutoField(primary_key=True)
    Date = models.DateTimeField()
    lastPrice = models.FloatField(default=0)
    lastVolume = models.FloatField(default=0)
    dayHigh= models.FloatField(default=0)
    dayLow = models.FloatField(default=0)
    open = models.FloatField(default=0)
    previousClose = models.FloatField(default=0)
    societe=models.ForeignKey(Societe, on_delete=models.CASCADE)
""" 
class Financials(models.Model):
    id=models.BigAutoField(primary_key=True)
    Date = models.DateTimeField()
    totalAssets = models.FloatField(default=0) #actifs totaux
    totalCurrentAssets = models.FloatField(default=0) #actifs courants totaux
    totalLiabilities = models.FloatField(default=0) #passifs totaux
    totalCurrentLiabilities = models.FloatField(default=0) #passifs courants totaux
    currentDebt = models.FloatField(default=0) #dette courante
    longTermDebt = models.FloatField(default=0) #dette à long terme (detteTotale= dette courante + dette à long terme)
    totalShareholderEquity = models.FloatField(default=0) #capitaux propres totaux
    netIncome = models.FloatField(default=0) #bénéfice net
    sellingGeneralAndAdministrative = models.FloatField(default=0) #ventes
    costofGoodsAndServicesSold = models.FloatField(default=0) #Coût des ventes 
    ebit = models.FloatField(default=0) #BAII (Bénéfice avant intérêts et impôts) 
    ebitda = models.FloatField(default=0) #Earnings Before Interest, Taxes, Depreciation and Amortization: EBITDA = bénéfice opérationnel + amortissements + dépréciations
    interestExpense = models.FloatField(default=0) #Charges d'intérêts
    investments = models.FloatField(default=0) # investissmeents
    commonStock = models.FloatField(default=0) # nombre d'actions ordinaires
    commonStockSharesOutstanding = models.FloatField(default=0) # nombre d'actions en circulation
    EPS = models.FloatField(default=0) # Earnings Per Stock (bénéfice par action BPA)
    dividendPayoutCommonStock = models.FloatField(default=0) # dividendes versés : la part des bénéfices de l'entreprise qui ont été distribués aux actionnaires sous forme de dividendes
    PERatio = models.FloatField(default=0) # (Price Earnings Ratio) : PER = prix de l'action / bénéfice par action
    PEGRatio = models.FloatField(default=0) # PEG (Price Earnings Growth) : PEG = PER / taux de croissance des bénéfices 
    PriceToBookRatio = models.FloatField(default=0) 
    ReturnOnAssetsTTM = models.FloatField(default=0) 
    ReturnOnEquityTTM = models.FloatField(default=0) 
    RevenueTTM = models.FloatField(default=0) 
    ProfitMargin = models.FloatField(default=0) # Marge bénéficiaire (rentabilité)
    Beta = models.FloatField(default=0) 
    societe=models.ForeignKey(Societe, on_delete=models.CASCADE)
 """