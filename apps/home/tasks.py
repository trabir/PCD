import yfinance as yf
from .models import Historical_data, Societe
from .models import RealTime_data
from django.utils import timezone
from celery import shared_task
import pandas as pd
import datetime
from django.db.models import Avg


@shared_task
def storeRealTimeData():#we store realtime data every 3 hours
    for soc in Societe.objects.all():
        actual_d = yf.Ticker(soc.label)
        act_d = RealTime_data(
                    Date=timezone.now(),
                    lastPrice=actual_d.fast_info["lastPrice"],
                    lastVolume=actual_d.fast_info["lastVolume"],
                    open=actual_d.fast_info["open"],
                    dayHigh=actual_d.fast_info["dayHigh"],
                    dayLow=actual_d.fast_info["dayLow"],
                    previousClose=actual_d.fast_info["previousClose"],
                    societe=soc,
                )
        act_d.save()

@shared_task
def deleteRealTimeData():#everyday at midnight all realtime data will be deleted
    my_models = RealTime_data.objects.all()
    all_companies=Societe.objects.all()
     # Group the real-time data by company and compute the mean for each attribute
    grouped_data = my_models.values('societe').annotate(high_mean=Avg('dayHigh'), volume_mean=Avg('lastVolume'), low_mean=Avg('dayLow'), open_mean=Avg('open'), close_mean=Avg('lastPrice'), previousClose=Avg('previousClose'))

    # Loop through each company in the grouped data and update the corresponding HistoricalData object
    for company_data in grouped_data:
        soc=all_companies.filter(id=company_data['societe'])[0]
        hist_d = Historical_data(
                Date=timezone.now(),
                Open=company_data['open_mean'],   
                High=company_data['high_mean'],
                Low=company_data['low_mean'],
                Close=company_data['close_mean'],
                Volume=company_data['volume_mean'],
                Rendement=(company_data['close_mean']-company_data['previousClose'])/company_data['previousClose'],
                societe=soc
            )
        hist_d.save()   
    RealTime_data.objects.all().delete()

     