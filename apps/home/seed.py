from apps.home.models import Secteur, Societe

def seed_dataSecteur():
    # create instances of MyModel
    sec=[
    Secteur(nom='Food'),
    Secteur(nom='Sports'),
    Secteur(nom='Banking'),
    Secteur(nom='Automotive'),
    Secteur(nom='Distribution'),
    Secteur(nom='Oil and Gas'),
    Secteur(nom='Technology and Computer Science'),
    Secteur(nom='Transportation'),
    Secteur(nom='Healthcare'),
    Secteur(nom='Manufacturing'),
    Secteur(nom='Education'),
    Secteur(nom='Trade')
    ]
    # Save the objects to the database in bulk
    Secteur.objects.bulk_create(sec)

def seed_dataSociete():
    # create instances of MyModel

    alimentaire = Secteur.objects.filter(nom='Food')
    sports = Secteur.objects.filter(nom='Sports')
    assurance = Secteur.objects.filter(nom='Banking')
    automobile = Secteur.objects.filter(nom='Automotive')
    distribution= Secteur.objects.filter(nom='Distribution')
    petrole_gaz = Secteur.objects.filter(nom='Oil and Gas')
    transport = Secteur.objects.filter(nom='Transportation')
    technologie_inf = Secteur.objects.filter(nom='Technology and Computer Science')
    med = Secteur.objects.filter(nom='Healthcare')
    indus = Secteur.objects.filter(nom='Manufacturing')
    edu = Secteur.objects.filter(nom='Education')
    tr = Secteur.objects.filter(nom='Trade')

    # Create a list of Societe objects
    soc=[
       Societe(label="KO", nom='Coca Cola', secteur=alimentaire[0]),
       Societe(label="PEP", nom='Pepsi', secteur=alimentaire[0]),
       Societe(label="KHC", nom='Kraft Heinz', secteur=alimentaire[0]),
       Societe(label="TSN", nom='Tyson Foods', secteur=alimentaire[0]),
       Societe(label="NKE", nom='Nike', secteur=sports[0]),
       Societe(label="UA", nom='Under Armour', secteur=sports[0]),
       Societe(label="RACE", nom='Ferrari N.V.', secteur=automobile[0]),
       Societe(label="VSTO", nom='Vista Outdoor', secteur=sports[0]),
       Societe(label="AEG", nom='Aegon', secteur=assurance[0]),
       Societe(label="MET", nom='MetLife', secteur=assurance[0]),
       Societe(label="TRV", nom='The Travelers Companies, Inc.', secteur=assurance[0]),
       Societe(label="AIG", nom='American International Group', secteur=assurance[0]),
       Societe(label="F", nom='Ford', secteur=automobile[0]),
       Societe(label="HMC", nom='Honda', secteur=automobile[0]),
       Societe(label="TSLA", nom='Tesla', secteur=automobile[0]),
       Societe(label="TM", nom='Toyota', secteur=automobile[0]),
       Societe(label="AMZN", nom='Amazon', secteur=distribution[0]),
       Societe(label="WMT", nom='Walmart', secteur=distribution[0]),
       Societe(label="DLTR", nom='Dollar Tree', secteur=distribution[0]),
       Societe(label="HD", nom='The Home Depot', secteur=distribution[0]),
       Societe(label="XOM", nom='Exxon Mobile', secteur=petrole_gaz[0]),
       Societe(label="COP", nom='ConocoPhillips', secteur=petrole_gaz[0]),
       Societe(label="PSX", nom='Phillips 66', secteur=petrole_gaz[0]),
       Societe(label="AAPL", nom='Apple', secteur=technologie_inf[0]),
       Societe(label="META", nom='Meta', secteur=technologie_inf[0]),
       Societe(label="MSFT", nom='Microsoft', secteur=technologie_inf[0]),
       Societe(label="UNP", nom='Union Pacific', secteur=transport[0]),
       Societe(label="FDX", nom='FedEx', secteur=transport[0]),
       Societe(label="DAR", nom='Delta Airlines', secteur=transport[0]),
       Societe(label="ABBV", nom='AbbVie', secteur=med[0]),
       Societe(label="ABM", nom='ABM Industries', secteur=indus[0]),
       Societe(label="ACN", nom='Accenture', secteur=indus[0]),
       Societe(label="ADNT", nom='Adient', secteur=indus[0]),
       Societe(label="ATGE", nom='Adtalem Global Education', secteur=edu[0]),
       Societe(label="6857.T", nom='Advantest', secteur=indus[0]),
       Societe(label="ACM", nom='Aecom', secteur=indus[0]),
       Societe(label="AER", nom='AerCap', secteur=transport[0]),
       Societe(label="AEM", nom='Agnico Eagle Mines Limited', secteur=tr[0]),
       Societe(label="APD", nom='Air Products and Chemicals', secteur=petrole_gaz[0]),
       Societe(label="AA", nom='Alcoa Corporation', secteur=indus[0]),
       Societe(label="ATUS", nom='Altice USA', secteur=technologie_inf[0]),
       Societe(label="MO", nom='Altria Group', secteur=indus[0]),
       Societe(label="AEE", nom='Ameren Corporation', secteur=petrole_gaz[0]),
       Societe(label="AEP", nom='American Electric Power Company', secteur=indus[0]),
       Societe(label="AMT", nom='American Tower Corporation', secteur=technologie_inf[0]),
       Societe(label="ABC", nom='AmerisourceBergen', secteur=med[0]),
       Societe(label="AME", nom='Ametek', secteur=indus[0]),
       Societe(label="AMR", nom='Alpha Metallurgical Resources', secteur=petrole_gaz[0]),
       Societe(label="CHGG", nom='Chegg, Inc.', secteur=edu[0]),
       Societe(label="AON", nom='Aon plc', secteur=assurance[0]),
       Societe(label="APTV", nom='Aptiv PLC', secteur=automobile[0]),
       Societe(label="MT", nom='ArcelorMittal', secteur=indus[0]),
       Societe(label="MU", nom='Micron Technology', secteur=technologie_inf[0]),
       Societe(label="ASH", nom='Ashland', secteur=indus[0]),
       Societe(label="AIZ", nom='Assurant', secteur=assurance[0]),
       Societe(label="T", nom='AT&T', secteur=technologie_inf[0]),
       Societe(label="AGR", nom='Avangrid', secteur=indus[0]),
       Societe(label="AVY", nom='Avery Dennison Corporation', secteur=indus[0]),
       Societe(label="CAR", nom='Avis Budget Group', secteur=automobile[0]),
       Societe(label="AVT", nom='Avnet', secteur=distribution[0]),
       Societe(label="BW", nom='Babcock & Wilcox Enterprises', secteur=indus[0]),
       Societe(label="IBA", nom='Industrias Bachoco', secteur=alimentaire[0]),
       Societe(label="BKR", nom='Baker Hughes Company', secteur=petrole_gaz[0]),
       Societe(label="BALL", nom='Ball Corporation', secteur=indus[0]),
       Societe(label="BBAR", nom='BBVA Argentina', secteur=assurance[0]),
       Societe(label="BHP", nom='BHP Group Limited', secteur=petrole_gaz[0]),
       Societe(label="BLK", nom='BlackRock', secteur=indus[0]),
       Societe(label="SQ", nom='Block', secteur=technologie_inf[0]),
       Societe(label="BA", nom='The Boeing Company', secteur=transport[0]),
       Societe(label="BCC", nom='Boise Cascade', secteur=tr[0]),
       Societe(label="BSX", nom='Boston Scientific', secteur=med[0]),
       Societe(label="BRC", nom='Brady Corporation', secteur=indus[0]),
       Societe(label="BCO", nom="The Brink's", secteur=indus[0]),
       Societe(label="BF-B", nom="Brown-Forman", secteur=alimentaire[0]),
       Societe(label="BBW", nom="Build-A-Bear Workshop", secteur=indus[0]),
       Societe(label="CBT", nom="Cabot Corporation", secteur=indus[0]),
       Societe(label="CCJ", nom="Cameco Corporation", secteur=indus[0]),
       Societe(label="CPB", nom="Campbell Soup Company", secteur=alimentaire[0]),
       Societe(label="CNQ", nom="Canadian Natural Resources Limited", secteur=petrole_gaz[0]),
       Societe(label="7751.T", nom="Canon", secteur=technologie_inf[0]),
       Societe(label="SBLK", nom="Star Bulk Carriers Corp", secteur=transport[0]),
       Societe(label="CAT", nom="Caterpillar", secteur=indus[0]),
       Societe(label="CBRE", nom="CBRE Group", secteur=assurance[0]),
       Societe(label="CLS", nom="Celestica", secteur=indus[0]),
       Societe(label="CVE", nom="Cenovus Energy", secteur=petrole_gaz[0]),
       Societe(label="CNP", nom="CenterPoint Energy", secteur=indus[0]),
       Societe(label="GIB", nom="CGI", secteur=technologie_inf[0]),
       Societe(label="CHPT", nom="ChargePoint Holdings", secteur=automobile[0]),
       Societe(label="CC", nom="The Chemours Company", secteur=indus[0]),
       Societe(label="CVX", nom="Chevron Corporation", secteur=petrole_gaz[0]),
       Societe(label="0941.HK", nom="China Mobile Limited", secteur=technologie_inf[0]),
       Societe(label="CMG", nom="Chipotle Mexican Grill", secteur=alimentaire[0]),
       Societe(label="CHH", nom="Choice Hotels International", secteur=tr[0]),
       Societe(label="CB", nom="Chubb Limited", secteur=tr[0]),
       Societe(label="CI", nom="The Cigna Group", secteur=assurance[0]),
       Societe(label="PCYG", nom="Park City Group", secteur=indus[0]),
       Societe(label="CIPLA.NS", nom="Cipla Limited", secteur=med[0]),
       Societe(label="C", nom="Citigroup", secteur=assurance[0]),
       Societe(label="CNHI", nom="CNH Industrial", secteur=indus[0]),
       Societe(label="CL", nom="Colgate-Palmolive Company", secteur=indus[0]),
       Societe(label="CMA", nom="Comerica", secteur=tr[0]),
       Societe(label="GLW", nom="Corning Incorporated", secteur=indus[0]),
       Societe(label="CTVA", nom="Corteva", secteur=indus[0]),
       Societe(label="CJR-B.TO", nom="Corus Entertainment", secteur=technologie_inf[0]),
       Societe(label="CPNG", nom="Coupang", secteur=tr[0]),
       Societe(label="CR", nom="Crane", secteur=indus[0]),
       Societe(label="CMI", nom="Cummins", secteur=automobile[0]),
       Societe(label="CVS", nom="CVS Health", secteur=indus[0]),
       Societe(label="CYIENT.NS", nom="Cyient Limited", secteur=technologie_inf[0]),
       Societe(label="APEI", nom="American Public Education, Inc.", secteur=edu[0]),
       Societe(label="DHR", nom="Danaher Corporation", secteur=technologie_inf[0]),
       Societe(label="DVA", nom="DaVita", secteur=indus[0]),
       Societe(label="DEO", nom="Diageo plc", secteur=alimentaire[0]),
       Societe(label="NRGV", nom="Energy Vault", secteur=indus[0]),
       Societe(label="ERF", nom="Enerplus", secteur=petrole_gaz[0]),
       Societe(label="ETR", nom="Entergy", secteur=indus[0]),
       Societe(label="EFX", nom="Equifax", secteur=indus[0]),
       Societe(label="EQNR", nom="Equinor", secteur=petrole_gaz[0]),
       Societe(label="EL", nom="The Estée Lauder", secteur=indus[0]),
       Societe(label="EB", nom="Eventbrite", secteur=indus[0]),
       Societe(label="EXC", nom="Exelon", secteur=assurance[0]),
       Societe(label="FDS", nom="FactSet", secteur=assurance[0]),
       Societe(label="DD", nom="DuPont de Nemours", secteur=indus[0]),
       Societe(label="EMN", nom="Eastman Chemical Company", secteur=indus[0]),
       Societe(label="ETN", nom="Eaton", secteur=indus[0]),
       Societe(label="EPC", nom="Edgewell Personal Care", secteur=indus[0]),
       Societe(label="EIX", nom="Edison International", secteur=indus[0]),
       Societe(label="LLY", nom="Eli Lilly and Company ", secteur=indus[0]),
       Societe(label="ERJ", nom="Embraer ", secteur=transport[0]),
       Societe(label="ENB", nom="Enbridge", secteur=transport[0]),
       Societe(label="ENR", nom="Energizer", secteur=indus[0]),
       Societe(label="FE", nom="FirstEnergy", secteur=indus[0]),
       Societe(label="FLT", nom="Fleetcor", secteur=tr[0]),
       Societe(label="FMC", nom="FMC Corporation", secteur=alimentaire[0]),
       Societe(label="DISH", nom="DISH Network", secteur=technologie_inf[0]),
       Societe(label="DLB", nom="Dolby Laboratories", secteur=technologie_inf[0]),
       Societe(label="DG", nom="Dollar General", secteur=transport[0]),
       Societe(label="PRU", nom="Prudential Financial, Inc.", secteur=assurance[0]),
       Societe(label="GHC", nom="Graham Holdings Co", secteur=edu[0]),
       Societe(label="FNMAL", nom="Federal National Mortgage Association", secteur=assurance[0]),
       Societe(label="FICO", nom="Fair Isaac", secteur=assurance[0]),
       Societe(label="FIS", nom="Fidelity National Information Services", secteur=assurance[0]),
       Societe(label="FL", nom="Foot Locker", secteur=alimentaire[0]),
       Societe(label="FORD", nom="Forward Industries", secteur=med[0]),
       Societe(label="LRN", nom="Stride Inc", secteur=edu[0]),
       Societe(label="FHL.SG", nom="Freddie Mac", secteur=assurance[0]),
       Societe(label="FYBR", nom="Frontier Communications", secteur=technologie_inf[0]),
       Societe(label="FRO", nom="Frontline", secteur=transport[0]),
       Societe(label="GCI", nom="Gannett", secteur=technologie_inf[0]),
       Societe(label="GIL", nom="Gildan Activewear", secteur=tr[0]),
       Societe(label="GLAXO.NS", nom="GlaxoSmithKline", secteur=med[0]),
       Societe(label="ASURB.MX", nom="Grupo Aeroportuario del Sureste", secteur=transport[0]),
       Societe(label="GES", nom="Guess", secteur=tr[0]),
       Societe(label="FUL", nom="H.B. Fuller", secteur=indus[0]),
       Societe(label="HAE", nom="Haemonetics", secteur=med[0]),
       Societe(label="HAL", nom="Halliburton", secteur=petrole_gaz[0]),
       Societe(label="HAS", nom="Hasbro", secteur=tr[0]),
       Societe(label="HDB", nom="HDFC Bank", secteur=assurance[0]),
       Societe(label="HLF", nom="Herbalife", secteur=tr[0]),
       Societe(label="HTHIY", nom="Hitachi", secteur=indus[0]),
       Societe(label="HON", nom="Honeywell", secteur=indus[0]),
       Societe(label="HRL", nom="Hormel", secteur=alimentaire[0]),
       Societe(label="HP", nom="Helmerich & Payne", secteur=petrole_gaz[0]),
       Societe(label="HSBC", nom="HSBC Holdings", secteur=assurance[0]),
       Societe(label="300345.SZ", nom="Hunan Huamin Holdings", secteur=technologie_inf[0]),
       Societe(label="HUBS", nom="HubSpot", secteur=technologie_inf[0]),
       Societe(label="HUM", nom="Humana", secteur=assurance[0]),
       Societe(label="HUN", nom="Huntsman", secteur=indus[0]),
       Societe(label="IAG", nom="IAMGOLD", secteur=indus[0]),
       Societe(label="IBM", nom="International Business Machines", secteur=technologie_inf[0]),
       Societe(label="INF.PA", nom="Infotel", secteur=technologie_inf[0]),
       Societe(label="INGR", nom="Ingredion", secteur=indus[0]),
       Societe(label="QQQ", nom="Invesco QQQ", secteur=assurance[0]),
       Societe(label="IQV", nom="IQVIA Holdings", secteur=med[0]),
       Societe(label="JEF", nom="Jefferies Financial", secteur=assurance[0]),
       Societe(label="JLL", nom="Jones Lang LaSalle Incorporated", secteur=indus[0]),     
       Societe(label="KBR", nom="KBR", secteur=technologie_inf[0]),
       Societe(label="KEY", nom="KeyCorp", secteur=assurance[0]),
       Societe(label="KEYS", nom="Keysight Technologies", secteur=technologie_inf[0]),
       Societe(label="KMB", nom="Kimberly-Clark Corporation ", secteur=indus[0]),
       Societe(label="KIM", nom="Kimco Realty", secteur=tr[0]),
       Societe(label="KSS", nom="Kohl's", secteur=distribution[0]),
       Societe(label="KR", nom="Kroger", secteur=tr[0]),
       Societe(label="KD", nom="Kyndryl", secteur=technologie_inf[0]),
       Societe(label="LVS", nom="Las Vegas Sands", secteur=tr[0]),
       Societe(label="LAZ", nom="Lazard", secteur=assurance[0]),
       Societe(label="LMND", nom="Lemonade", secteur=assurance[0]),
       Societe(label="LEN", nom="Lennar", secteur=indus[0]),
       Societe(label="LPL", nom="LG Display", secteur=tr[0]),
       Societe(label="LITB", nom="LightInTheBox", secteur=tr[0]),
       Societe(label="LOW", nom="Lowe's", secteur=distribution[0]),
       Societe(label="LUMN", nom="Lumen Technologies", secteur=technologie_inf[0]),
       Societe(label="LYB", nom="LyondellBasell", secteur=indus[0]),
       Societe(label="MTB", nom="M&T Bank", secteur=assurance[0]),
       Societe(label="M", nom="Macy's", secteur=indus[0]),
       Societe(label="MNK", nom="Mallinckrodt", secteur=med[0]),
       Societe(label="LAUR", nom="Laureate Education, Inc.", secteur=edu[0]),
       Societe(label="MPC", nom="Marathon Petroleum", secteur=petrole_gaz[0]),
       Societe(label="MA", nom="Mastercard", secteur=assurance[0]),
       Societe(label="MAT", nom="Mattel", secteur=indus[0]),
       Societe(label="MCK", nom="McKesson", secteur=med[0]),
       Societe(label="MEG", nom="Montrose Environmental", secteur=indus[0]),
       Societe(label="MUFG", nom="Mitsubishi UFJ Financial ", secteur=assurance[0]),
       Societe(label="MHK", nom="Mohawk Industries", secteur=indus[0]),
       Societe(label="MNST", nom="Monster Beverage", secteur=alimentaire[0]),
       Societe(label="MCO", nom="Moody's", secteur=assurance[0]),
       Societe(label="MSI", nom="Motorola Solutions", secteur=technologie_inf[0]),
       Societe(label="MOV", nom="Movado", secteur=distribution[0]),
       Societe(label="MSCI", nom="MSCI", secteur=assurance[0]),
       Societe(label="NEWR", nom="New Relic", secteur=technologie_inf[0]),
       Societe(label="9432.T", nom="Nippon Telegraph and Telephone", secteur=technologie_inf[0]),
       Societe(label="NI", nom="NiSource", secteur=transport[0]),
       Societe(label="NOK", nom="Nokia", secteur=technologie_inf[0]),
       Societe(label="JWN", nom="Nordstrom", secteur=tr[0]),
       Societe(label="NVS", nom="Novatris", secteur=med[0]),
       Societe(label="NUE", nom="Nucor", secteur=indus[0]),
       Societe(label="CL=F", nom="Crude Oil May 23", secteur=petrole_gaz[0]),
       Societe(label="OKE", nom="ONEOK", secteur=transport[0]),
       Societe(label="IX", nom="Orix", secteur=indus[0]),
       Societe(label="OVV", nom="Ovintiv", secteur=petrole_gaz[0]),
       Societe(label="OC", nom="Owens Corning", secteur=indus[0]),
       Societe(label="OXUR.BR", nom="Oxurion", secteur=med[0]),
       Societe(label="GM", nom="General Motors Company", secteur=automobile[0]),
       Societe(label="PFE", nom="Pfizer", secteur=med[0]),
       Societe(label="0857.HK", nom="PetroChina", secteur=petrole_gaz[0]),
       Societe(label="PKI", nom="PerkinElmer", secteur=technologie_inf[0]),
       Societe(label="PHG", nom="Philips", secteur=indus[0]),
       Societe(label="P4F.SG", nom="Pinnacle Foods", secteur=alimentaire[0]),
       Societe(label="PXD", nom="Pioneer Natural Resources", secteur=petrole_gaz[0]),
       Societe(label="PHI", nom="PLDT", secteur=technologie_inf[0]),
       Societe(label="QGEN", nom="Qiagen", secteur=indus[0]),
       Societe(label="RTX", nom="Raytheon", secteur=indus[0]),
       Societe(label="RSG", nom="Republic Services", secteur=indus[0]),
       Societe(label="RBA", nom="Ritchie Bros. Auctioneers", secteur=tr[0]),
       Societe(label="RIO", nom="Rio Tinto", secteur=indus[0]),
       Societe(label="RPM", nom="RPM International", secteur=indus[0]),
       Societe(label="RGR", nom="Sturm, Ruger", secteur=indus[0]),
       Societe(label="SPGI", nom="S&P Global", secteur=assurance[0]),
       Societe(label="SNY", nom="Sanofi", secteur=med[0]),
       Societe(label="SXT", nom="Sensient", secteur=indus[0]),
       Societe(label="NOW", nom="ServiceNow", secteur=technologie_inf[0]),
       Societe(label="SHW", nom="Sherwin-Williams", secteur=indus[0]),
       Societe(label="SSTK", nom="Shutterstock", secteur=indus[0]),
       Societe(label="SIE.DE", nom="Siemens", secteur=indus[0]),
       Societe(label="SHIIY", nom="Sinopec", secteur=indus[0]),
       Societe(label="SIX", nom="Six Flags", secteur=tr[0]),
       Societe(label="SLB", nom="Schlumberger Limited", secteur=petrole_gaz[0]),
       Societe(label="SMAR", nom="Smartsheet", secteur=technologie_inf[0]),
       Societe(label="SWBI", nom="Smith & Wesson Brands", secteur=indus[0]),
       Societe(label="SNAP", nom="Snap", secteur=technologie_inf[0])
    ]

    # Save the objects to the database in bulk
    Societe.objects.bulk_create(soc)