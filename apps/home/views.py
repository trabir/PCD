from django import template
from django.contrib.auth.decorators import login_required
import numpy as np
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render, redirect
from django.template import loader
from django.urls import reverse
import yfinance as yf
import pandas as pd
from .models import Historical_data, Secteur, Societe
from apps.authentication.forms import UpdateUserForm, UpdateUserProfileForm
import json
from datetime import datetime, timedelta
from io import BytesIO
import base64
from pytz import timezone


API_KEY = '4VZJI103X9MJ1XKP'


@login_required(login_url="/login/")
def index(request):
    html_template = loader.get_template('home/index.html')
    storeHistoricalData()
    res = loadSocietes()
    data, momentaneite = calcul_rendement_strategie_momentaneite()

    # DecisionTreeRegressorFor_PricePrediction(res[0])
    # loadFinancials()
    classification_for_PricePrediction(data, momentaneite)
    context = {'data': regression_for_PricePrediction(res[0], momentaneite), 'secteurs': loadSecteurs(
    ), 'societes': res, 'all_soc': Societe.objects.all().values()}
    return HttpResponse(html_template.render(context, request))


def loadSecteurs():
    secteurs = Secteur.objects.all().order_by('nom').values()
    return secteurs


def loadSocietes():
    automobile = Secteur.objects.filter(nom='Automotive')[0]
    societes = Societe.objects.filter(secteur=automobile).order_by('label')
    return societes


def update_selected_secteur(request):
    selected_value = request.GET.get('selected_value')
    sect = Secteur.objects.filter(nom=selected_value)[0]
    societes = Societe.objects.filter(secteur=sect).order_by('label')
    res = societes.values('label', 'nom')
    data, momentaneite = calcul_rendement_strategie_momentaneite()
    return JsonResponse({'options': list(res), 'hist': regression_for_PricePrediction(societes[0], momentaneite)})


""" def loadHistoricalData(soc):
    # using Django ORM to load all the historical data from PostgreSQL Database
    historical_data = Historical_data.objects.filter(societe=soc)
    # we converted the queryset to a list of dictionaries using the values() method
    hd = list(historical_data.values())
   # we onverted the list of dictionaries to a JSON string using the json.dumps() method
    hd_json = json.dumps(hd, indent=4, default=str)
    return hd_json """


def loadFinancials():
    fd = FundamentalData(API_KEY, output_format='pandas')
    balance_sheet_data, _ = fd.get_company_overview('NKE')
    print(balance_sheet_data.columns)


def calcul_rendement_strategie_momentaneite():
    # Extraire les données historiques de la table Historical_data
    historical_data = Historical_data.objects.all().order_by('Date').values(
        'Date', 'Close', 'Volume', 'High', 'Rendement', 'Rendement_moyen')

    # Conversion des données en DataFrame Pandas
    data = pd.DataFrame(list(historical_data.values()))

    # Group the data by date
    grouped_data = data.groupby('Date')

    # Initialize an empty list to store the filtered data for each date
    ensemble_sup = []
    ensemble_inf = []

    # Iterate over the groups and filter the data for each date
    for date, group in grouped_data:
        sup_group = group[group['Rendement'] >= group['Rendement_moyen']]
        ensemble_sup.append(sup_group)
        inf_group = group[group['Rendement'] < group['Rendement_moyen']]
        ensemble_inf.append(inf_group)

    # Concatenate the filtered data for each date into a single dataframe
    ensemble_sup = pd.concat(ensemble_sup)
    ensemble_inf = pd.concat(ensemble_inf)

    # Calculer la moyenne pour chaque ensemble: Assuming ensemble_sup and ensemble_inf are  pandas dataframes with columns 'Date' and 'Rendement'
    moyenne_sup = ensemble_sup.groupby('Date')['Rendement'].mean()
    moyenne_inf = ensemble_inf.groupby('Date')['Rendement'].mean()

    # Calculer la différence de moyenne qui va donner le rendement de la stratégie de momentanéité
    moyenne_strategie = moyenne_sup - moyenne_inf
    print(moyenne_strategie)
    moyenne_strategie = moyenne_strategie.rename_axis(
        'Date').reset_index(name='Rendement_strategie')
    
    #pd.to_numeric
    return data, moyenne_strategie


def classification_for_PricePrediction(data, moy_s):
    # merge the two dataframes on Date column
    merged_data = pd.merge(data, moy_s, on='Date')
    merged_data = merged_data[pd.to_datetime(
        merged_data['Date']).dt.date == pd.to_datetime('2023-04-05').date()]

    merged_data['Rend_Sup'] = merged_data.apply(
        lambda x: True if x['Rendement'] > x['Rendement_moyen'] else False, axis=1)

    # Create a new DataFrame with the relevant features
    X = merged_data[['Rendement', 'Rendement_moyen', 'Rend_Sup']]

    # Set the target variable
    y = merged_data['Rendement'] > merged_data['Rendement_moyen']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=10)

    # Create the SVM model and fit it to the training data
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # Predict the stock prices using the SVM model
    y_pred = svm.predict(X_test)

    # Calculer l'exactitude
    accuracy = accuracy_score(y_test, y_pred)
    print("Exactitude : ", accuracy)

    # Calcul de la précision
    precision = precision_score(y_test, y_pred)
    print("Précision : ", precision)

    # Evaluate the model using the R-squared metric
    score = svm.score(X_test, y_test)
    print(f"R-squared score: {score}")


def regression_for_PricePrediction(soc, mo):
    
    # Extraire les données historiques de la table Historical_data
    historical_data = Historical_data.objects.filter(societe=soc).order_by(
        'Date').values('Date', 'Close', 'Volume', 'High', 'Rendement', 'Rendement_moyen')
    length = historical_data.count()

    # Conversion des données en DataFrame Pandas
    data = pd.DataFrame(list(historical_data.values()))
    data = pd.merge(data, mo, on='Date')
    print
    data['day'] = np.arange(1, length+1)

    # Calcul du prix de clôture précédent car nous ne pouvons pas utiliser les valeurs de close telles quelles, car elles sont déjà associées à la date correspondante et ne peuvent pas être utilisées comme variables indépendantes pour la prédiction des jours suivants. Ainsi, nous décalons les valeurs de close d'un jour vers le haut (c'est-à-dire la première valeur devient la deuxième, la deuxième devient la troisième, etc.) afin de les utiliser comme variables indépendantes. Cela signifie que la valeur de close pour le 6ème jour sera utilisée comme variable dépendante (ou target) pour l'entraînement du modèle.
    data['Close_shifted'] = data['Close'].shift(1)

    # Suppression de la première ligne (qui contient une valeur NaN)
    data = data.dropna()

    # Séparation des données en données d'entraînement et de test
    # variables indépendantes
    X = data[['day', 'High', 'Volume', 'Rendement_strategie']]
    # variable dépendante que je cherche à prédire dans le futur
    Y = data['Close']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Entraînement du modèle de régression linéaire
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)

    # Calcul de l'erreur quadratique moyenne
    error = mean_squared_error(y_test, y_pred)

    # Renvoyer le coefficient de détermination (R²) pour évaluer la qualité du modèle
    score = regressor.score(x_test, y_test)

    # Prédiction des prix de clôture pour les 10 prochains jours
    next_10_days = pd.DataFrame({
        'day': np.arange(length+1, length+11),
        # la moyenne répétée sur 10 lignes
        'High': [data['High'].mean()] * 10,  # la moyenne répétée sur 10 lignes
        # la moyenne répétée sur 10 lignes
        'Volume': [data['Volume'].mean()] * 10,
        'Rendement_strategie': [data['Rendement_strategie'].mean()] * 10
    })

    # Faire des prédictions pour les 10 prochains jours
    next_10_days['Close_predicted'] = regressor.predict(next_10_days)

    # print(next_10_days[['Close_predicted']])
    start_date = datetime.now().date() + timedelta(days=1)
    end_date = start_date + timedelta(days=9)

    date_range = pd.date_range(start=start_date, end=end_date)
    datetime_range = date_range.to_pydatetime().tolist()

    result = pd.DataFrame({
        'Date': datetime_range,
        'Close_predicted': next_10_days['Close_predicted'].to_list()})
    dict_res = result.to_dict('records')

    x_train['Date'] = x_train.apply(
        lambda row: data['Date'][row['day']-1], axis=1)
    x_test['Date'] = x_test.apply(
        lambda row: data['Date'][row['day']-1], axis=1)
    x_test['testRes'] = y_pred
    x_train = x_train.sort_values(by="Date")
    x_test = x_test.sort_values(by="Date")

    final_res = {
        'prediction': dict_res,
        'coeff_d': score,
        'x_train': x_train.to_dict('records'),
        'y_train': y_train.to_list(),
        'x_test': x_test.to_dict('records'),
        'y_test': y_test.to_list(),
        "dates": data['Date'].to_list()
    }
    return (json.dumps(final_res,  indent=4, default=str))


def DecisionTreeRegressorFor_PricePrediction(soc):
    # Extraire les données historiques de la table Historical_data
    historical_data = Historical_data.objects.filter(societe=soc).order_by(
        '-Date').values('Date', 'Close', 'Volume', 'High', 'Low')
    length = historical_data.count()

    # Conversion des données en DataFrame Pandas
    data = pd.DataFrame(list(historical_data.values()))

    data['day'] = np.arange(1, length+1)

    # Calcul du prix de clôture précédent car nous ne pouvons pas utiliser les valeurs de close telles quelles, car elles sont déjà associées à la date correspondante et ne peuvent pas être utilisées comme variables indépendantes pour la prédiction des jours suivants. Ainsi, nous décalons les valeurs de close d'un jour vers le haut (c'est-à-dire la première valeur devient la deuxième, la deuxième devient la troisième, etc.) afin de les utiliser comme variables indépendantes. Cela signifie que la valeur de close pour le 6ème jour sera utilisée comme variable dépendante (ou target) pour l'entraînement du modèle.
    data['Close_shifted'] = data['Close'].shift(1)

    # Suppression de la première ligne (qui contient une valeur NaN)
    data = data.dropna()

    # Séparation des données en données d'entraînement et de test
    # variables indépendantes
    X = data[['day', 'Close_shifted', 'High', 'Low', 'Volume']]
    # variable dépendante que je cherche à prédire dans le futur
    Y = data['Close']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Instanciation du modèle d'arbre de régression
    tree_reg = DecisionTreeRegressor(max_depth=3)

    # Entraînement du modèle sur l'ensemble d'entraînement
    tree_reg.fit(x_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = tree_reg.predict(x_test)

    error = mean_squared_error(y_test, y_pred)
    # Renvoyer le coefficient de détermination (R²) pour évaluer la qualité du modèle
    score = tree_reg.score(x_test, y_test)

    # print(score)


def onChangeHistoricalData(request):
    selected_value = request.GET.get('selected_valuee')
    data, momentaneite = calcul_rendement_strategie_momentaneite()
    soc = Societe.objects.filter(label=selected_value)[0]
    res = regression_for_PricePrediction(soc, momentaneite)
    return JsonResponse({'res': res})


""" def onChangeHistoricalData(request):
    selected_value = request.GET.get('selected_valuee')
    # using Django ORM to load all the historical data from PostgreSQL Database
    historical_data = Historical_data.objects.filter(
        societe=Societe.objects.filter(label=selected_value)[0])
    # we converted the queryset to a list of dictionaries using the values() method
    hd = list(historical_data.values())
    # we onverted the list of dictionaries to a JSON string using the json.dumps() method
    hd_j = json.dumps(hd, indent=4, default=str)

    return JsonResponse({'res': hd_j}) """


def storeHistoricalData():  # we execute this method only once to retrieve the historical data  from 20-01-2023 up to the day of the first execution app that's why we check whether the table is empty or not
    if Historical_data.objects.exists() == False:
        for soc in Societe.objects.all():
            # import historical data from yahoo finance
            historical_d = yf.Ticker(soc.label)
            START = pd.to_datetime('2023-01-15')
            END = pd.to_datetime('today') - timedelta(days=1)
            data_from_15_01 = historical_d.history(start=START, end=END)

            # we import the row which has a date just before 20-01-2023 for calculating "rendement" because the formula depends on the price of last day
            eastern_tz = timezone('America/New_York')
            start_ts = pd.Timestamp(
                pd.to_datetime('2023-01-20'), tz=eastern_tz)
            # data entre [20-01.. hier] hier par rapport à la date système lorsqu'on execute l'app pour la première fois
            data_from_20_01 = data_from_15_01.loc[data_from_15_01.index >= start_ts]
            # data entre [15-01 ET 20-01]
            data_before_20_01 = data_from_15_01.loc[data_from_15_01.index < start_ts]

            # get the last row in data_before_20_01 which represents the date_just_before_20_01
            date_just_before_20_01 = data_before_20_01.iloc[[-1]]
            # Convert the date index to a separate column
            data_from_15_01.reset_index(inplace=True)
            data_from_20_01.reset_index(inplace=True)

            for index, row in data_from_20_01.iterrows():  # to respect the MVC architecture
                if (index == 0):
                    hist_d = Historical_data(
                        Date=row["Date"],
                        Open=row["Open"],
                        High=row["High"],
                        Low=row["Low"],
                        Close=row["Close"],
                        Volume=row["Volume"],
                        Rendement=(row["Close"]-date_just_before_20_01['Close']
                                   [0])/date_just_before_20_01['Close'][0],
                        societe=soc
                    )
                else:
                    hist_d = Historical_data(
                        Date=row["Date"],
                        Open=row["Open"],
                        High=row["High"],
                        Low=row["Low"],
                        Close=row["Close"],
                        Volume=row["Volume"],
                        Rendement=(row["Close"]-data_from_20_01.iloc[index-1]
                                   ['Close'])/data_from_20_01.iloc[index-1]['Close'],
                        societe=soc
                    )
                hist_d.save()

    data = pd.DataFrame(list(Historical_data.objects.all().values()))
    # le rendement moyen de toutes les actions(sociétés) chaque jour
    daily_mean = data.groupby('Date')['Rendement'].mean()

    # Iterate over each date and update the daily_mean attribute
    for date, mean in daily_mean.items():
        objects = Historical_data.objects.filter(Date=date)
        objects.update(Rendement_moyen=mean)


@login_required(login_url="/login/")
def user(request):
    encoded_data = None
    binary_data = None
    if (request.user.userprofile.avatar):
        binary_data = request.user.userprofile.avatar.tobytes()

    if request.method == "POST":
        u_form = UpdateUserForm(request.POST, instance=request.user)
        p_form = UpdateUserProfileForm(request.POST, request.FILES,
                                       instance=request.user.userprofile)
        if u_form.is_valid() and p_form.is_valid():
            if(request.FILES and request.FILES['avatar']):
                binary_data = request.FILES['avatar'].open().read()
            u_form.save()
            user_profile = p_form.save(commit=False)
            user_profile.avatar = binary_data
            user_profile.save()
            return redirect("user.html")

    else:
        u_form = UpdateUserForm(instance=request.user)
        p_form = UpdateUserProfileForm(instance=request.user.userprofile)
        if(request.user.userprofile.avatar):
            image_bytes = BytesIO(request.user.userprofile.avatar).read()
            encoded_data = base64.b64encode(image_bytes).decode('utf-8')

    context = {
        'u_form': u_form,
        'p_form': p_form,
        'encodedBinary': encoded_data
    }

    return render(request, "home/user.html", context)


def navig_profilePicture(request):
    encoded_data = None
    if(request.user.userprofile.avatar):
        image_bytes = BytesIO(request.user.userprofile.avatar).read()
        encoded_data = base64.b64encode(image_bytes).decode('utf-8')
    return ({'encodedData': encoded_data})


def delete_avatar(request):
    profile = request.user.userprofile
    profile.avatar = None
    profile.save()
    # return redirect('user')
    return JsonResponse({'success': True})


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))
