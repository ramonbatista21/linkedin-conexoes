#!/usr/bin/env python
# coding: utf-8

# # Mineiração de Dadosdo LinkedIn

# ## Obtenção dos dados

# In[1]:


#Os dados são obtidos nas configurações do seu perfil no LinkedIn

from IPython.display import Image
Image(filename='img/linkedIn.png')


# #### As informações de nome foram alteradas para preservar a privacidade dos meus contatos. Portanto os nomes utilizados nessa análise são fakes

# In[2]:


#Importação das bibliotecas

#criação de fakers com os names
from faker import Faker 

#Manipulação de datas
import datetime 

#Manipulação de dados
import pandas as pd 

#Manipulação de dados
import numpy as np 

#Criação de Gráficos
import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#Localização
from geopy import geocoders
import reverse_geocoder as rg

#Mineração de textos
from nltk.metrics.distance import edit_distance
import nltk
from nltk.metrics.distance import jaccard_distance

#Aprendizado de máquina não supervisionado - Clustering
from sklearn.cluster import KMeans
import simplekml


# In[3]:


#Importação dos dados

conexoes = pd.read_csv('datasets/conect_fake.csv') #Já utilizando dados fakes

conexoes.head()


# In[4]:


#Criando variável

fake = Faker()


# In[5]:


#Verificando a criação do Primeiro Nome

fake.first_name()


# In[6]:


#Verificando a criação do último Nome

fake.last_name()


# In[7]:


#Verificando a criação do E-mail

fake.email()


# In[8]:


#Resetando index

conexoes = conexoes.reset_index(drop=True)


# In[9]:


#Criando Nomes Fakes

for i in range(0, len(conexoes)):
    conexoes['First Name'][i] = fake.first_name()

conexoes.head()


# In[10]:


#Criando Sobrenomes Fakes

for i in range(0, len(conexoes)):
    conexoes['Last Name'][i] = fake.last_name()

conexoes.head()


# In[11]:


#Criando E-mails Fakes

for i in range(0, len(conexoes)):
    conexoes['Email Address'][i] = fake.email()

conexoes.head()


# In[12]:


#Exportando dataset com fake

conexoes.to_csv('datasets/conect_fake.csv', index = False)


# In[13]:


#Nova coluna com nome e sobrenome

conexoes['Full Name'] = conexoes['First Name'] + ' ' + conexoes['Last Name']
conexoes.head()


# In[14]:


#Removendo colunas desnecessárias

conexoes.drop(columns=['First Name','Last Name','Email Address'], axis=1, inplace=True)
conexoes.head()


# In[15]:


#Resetando index

conexoes.reset_index(drop=True, inplace=True)
conexoes.head()


# In[16]:


#Convertendo datas

conexoes['Connected On'] = pd.to_datetime(conexoes['Connected On'])
conexoes.head()


# In[17]:


#Gráfico Nome x Ano

grafico = px.scatter(conexoes, x = 'Full Name', y = 'Connected On', labels={
                     "Full Name": "Nome",
                     "Connected On": "Período",
                 }, 
                    title='Novas conexão por período')

grafico.show()


# In[18]:


#Gráfico Nome x Ano

grupo = conexoes.groupby(by = 'Connected On').count()


# In[19]:


#Gráfico Quantidade de conexões por data

grafico = px.line(grupo, 
                  title = 'Novas Conexões por data',
                 labels={ "value": "Valor",
                     "Connected On": "Período",
                        'variable': 'Categoria'})
grafico.show()


# In[20]:


#Criando coluna somente com o ano

conexoes['ano'] = (conexoes['Connected On'].dt.year)
conexoes['ano'].head()


# In[21]:


#Verificando valores na

conexoes.isna().sum()


# In[22]:


#Removendo valores duplicados

conexoes.drop_duplicates()


# In[23]:


#Removendo valores nulos

conexoes.dropna(inplace=True)
conexoes.isna().sum()


# In[24]:


#Convertendo ano em inteiro

conexoes['ano'] = conexoes['ano'].astype(int)


# In[25]:


#Criando coluna com mês 

conexoes['mes'] = (conexoes['Connected On'].dt.month_name())
conexoes['mes'].head()


# In[26]:


conexoes['mes_n'] = (conexoes['Connected On'].dt.month)
conexoes['mes_n'].head()


# In[27]:


#Agrupando por mês

grupo_mes = conexoes.groupby(by = 'mes').count()
grupo_mes


# In[28]:


#Grafico Quantidade de conexões por mês

grafico = px.line(grupo_mes,
                  title = 'Novas conexões por mês', 
                 labels={ "value": "Valor",
                     "mes": "Mês",
                        'variable': 'Categoria'})

grafico.show()


# In[29]:


#Ordedando os meses

conexoes.sort_values(by=['mes_n'], inplace=True)

conexoes


# In[30]:


#Tradução dos meses
conexoes['mes'] = conexoes['mes'].replace({'January': 'Janeiro',
                                          'February': 'Fevereiro',
                                          'March': 'Março',
                                          'April': 'Abril',
                                           'May': 'Maio',
                                           'June': 'Junho',
                                           'July': 'Julho',
                                           'August': 'Agosto',
                                           'September': 'Setembro',
                                           'October': 'Outubro',
                                           'November': 'Novembro',
                                           'December': 'Dezembro',})


# In[31]:


#Contagem por mês 
grafico = px.histogram(conexoes['mes'], 
                       title = 'Novas conexões por mês',
                       labels={ "value": "Mês",
                         'variable': 'Categoria',})

grafico.show()


# In[32]:


#Contagem por ano
grupo_ano = conexoes['ano'].value_counts()
grupo_ano


# In[33]:


#Resetando os índices
grupo_ano = grupo_ano.reset_index()


# In[34]:


#Renomeando as colunas
grupo_ano.rename(columns={'index': 'ano',
                          'ano': 'Quantidade'}, inplace=True)


# In[35]:


#Ordenando por ano
grupo_ano.sort_values('ano', inplace=True)


# In[36]:


#Grafico contagem por ano
grafico = px.bar(x = grupo_ano['ano'], y = grupo_ano['Quantidade'],  
                 title = 'Novas Conexões por ano',
                       labels={ "y": "Quantidade",
                         'x': 'Ano',})

grafico.show()


# In[37]:


#Valores unicos
np.unique(conexoes['Company'], return_counts=True)


# In[38]:


#Tamanho dataset
conexoes.shape


# In[39]:


#Quantidade de valores unicos
len(np.unique(conexoes['Company']))


# In[40]:


#Treemapping por empresa

grafico = px.treemap(conexoes, path=['Company', 'Position', 'Full Name'])
grafico.show()


# In[41]:


#Conferindo valores unicos
np.unique(conexoes['Company'], return_counts=True)


# In[42]:


#Conferindo valores unicos

np.unique(conexoes['Position'], return_counts=True)


# In[43]:


#Conferindo valores unicos
conexoes['Company'].shape


# In[44]:


#Quantidade de cargos
len(np.unique(conexoes['Position']))


# In[45]:


#Histograma de cargos
grafico = px.histogram(conexoes['Position'],  
                 title = 'Quantidade de Cargos',
                       labels={ "value": "Cargo",
                         'variable': 'Categoria',})
grafico.show()


# In[46]:


#Treemapping por cargo

grafico = px.treemap(conexoes, path = ['Position', 'Company', 'Full Name'])

grafico.show()


# # Agrupamento dos Cargos

# In[47]:


#Entendendo os bigramas
bigramas_c1 = nltk.bigrams('analista de business intellicenge')


# In[48]:


'analista de business inteligeence'.split()


# In[49]:


bigramas_c1 = list(nltk.bigrams('analista de business intelligence'.split(), pad_right=True, pad_left=True ))


# In[50]:


list(bigramas_c1)


# In[51]:


bigramas_c2 = list(nltk.bigrams('cientista de dados'.split(), pad_right=True, pad_left=True ))

bigramas_c2


# In[52]:


bigramas_c3 = list(nltk.bigrams('analista de dados'.split(), pad_right=True, pad_left=True ))

bigramas_c3


# ### Jaccard

# In[53]:


c1 = 'analista de business intelligence'.split()
c2 = 'cientista de dados'.split()
c3 = 'analista de dados'.split()


# In[54]:


c1, c2, c3


# In[55]:


#c1 x c3

intersecao = set(c1).intersection(set(c3))
intersecao


# In[56]:


uniao = set(c1).union(set(c3))
uniao


# In[57]:


(len(uniao) - len(intersecao)) / len(uniao)


# In[58]:


#c1 x c1

jaccard_distance(set(c1), set(c1))


# In[59]:


#c1 x c3

jaccard_distance(set(c1), set(c3))


# In[60]:


#Criando um dataframe de teste
cargos_df = pd.DataFrame(columns=['Position'], data=['Estagiário',
                                                     'Estagiário de Engenharia',
                                                     'Assitente Administrativo',
                                                     'Auxiliar Administrativo'])

cargos_df


# In[61]:


#Retornando um array

todos_cargos = cargos_df['Position'].values

todos_cargos


# In[62]:


set(todos_cargos)


# In[63]:


#Aplicando a função de distancia de jaccard no dataframe de teste
for cargo1 in todos_cargos:
    print(cargo1)
    print('------------')
    for cargo2 in todos_cargos:
        print(cargo2, jaccard_distance(set(cargo1), set(cargo2)))
    print('\n')
    
#Quanto menor a distância, mais próxima são as palavras


# In[64]:


#Criando os clusters com os cargos
limite = 0.3
clusters = {}
for cargo1 in todos_cargos:
    clusters[cargo1]=[]
    for cargo2 in todos_cargos:
        if cargo2 in clusters[cargo1] or cargo2 in clusters and cargo1 in clusters[cargo2]:
            continue
            
        distancia = jaccard_distance(set(cargo1), set(cargo2))
        
        if distancia <= limite:
            clusters[cargo1].append(cargo2)


# In[65]:


#Verificando os clusters
clusters


# In[66]:


#Retornando os clusters com mais de um cargo

cluster = [clusters[cargo] for cargo in clusters if len(clusters[cargo]) > 1]


# In[67]:


#Verificando os clusters

cluster


# In[68]:


#Aplicando no dataset completo
todos_cargos = conexoes['Position'].values
len(todos_cargos)


# In[69]:


#Criando os sets
todos_cargos = set(todos_cargos)


# In[70]:


#Tamanho do set
len(todos_cargos)


# In[71]:


#Criando os clusters com os cargos

limite = 0.1
clusters = {}
for cargo1 in todos_cargos:
    clusters[cargo1]=[]
    for cargo2 in todos_cargos:
        if cargo2 in clusters[cargo1] or cargo2 in clusters and cargo1 in clusters[cargo2]:
            continue
            
        distancia = jaccard_distance(set(cargo1), set(cargo2))
        
        if distancia <= limite:
            clusters[cargo1].append(cargo2)


# In[72]:


#Retornando os clusters com mais de um cargo

clusters = [clusters[cargo] for cargo in clusters if len(clusters[cargo]) > 1]


# In[73]:


clusters, len(clusters)


# In[74]:


#Resetando index
conexoes.reset_index(drop=True, inplace=True)


# In[75]:


#Criando um dicionario com os nomes
cluster_contato = {}
for cluster in clusters:
    #print(cluster)
    cluster_contato[tuple(cluster)] = []
    for contato in range(0, len(conexoes)):
        if conexoes['Position'][contato] in cluster:
            cluster_contato[tuple(cluster)].append(conexoes['Full Name'][contato])

cluster_contato


# In[76]:


#Criando uma visualização em html

from IPython.core.display import HTML

for cargos in cluster_contato:
    lista_cargos = 'Lista de cargos no grupos: ' + ', '.join(cargos)
    #print(lista_cargos)
    
    termos = set(cargos[0].split())
                 
    for palavras in cargos:
        termos.intersection_update(set(palavras.split()))
    if len(termos)== 0:
        termos = ['*** Nenhum termo em comum ***']
    termos_impressao = 'Termos Comuns:' + ', '.join(termos)
    
    display(HTML(f'<h5>{lista_cargos}</h3>'))
    display(HTML(f'<p>{termos_impressao}</p>'))
    display(HTML(f'<p>{"-"*70}</p>'))
    display(HTML(f'<p><mark>{", ".join(cluster_contato[cargos])}</mark></p>'))


# In[77]:


#Trabalhado com json e html
import json
import codecs

from IPython.core.display import display


# In[78]:


#Salvando em um arquivo json
saida_json = {'name': 'LinkedIn', 'children': []}

for grupos in cluster_contato:
    #print(grupos)
    saida_json['children'].append({'name': ',' .join(grupos)[:20],
                                  'children': [{'name': contato} for contato in cluster_contato[grupos]]})
    
    f = open('json/dadoslink.json', 'w', encoding='utf8')
    f.write(json.dumps(saida_json, indent=1, ensure_ascii=False))
    f.close


# In[79]:


#Abrindo o json
data = json.load(open('json/dadoslink.json', encoding='utf8'))

data


# In[80]:


#Retornando os valres
valores = json.dumps(data, ensure_ascii=False)
valores


# In[81]:


#Criando um novo json
saida_json = {'name': 'Linkedin', 'children': []}
contador = 1
for grupos in cluster_contato:
  #print(grupos)
  if contador > 10:
    break
  
  saida_json['children'].append({'name': ', '.join(grupos)[:20],
                                 'children': [{'name': contato} for contato in cluster_contato[grupos]]})
  f = open('json/dados2.json', 'w')
  f.write(json.dumps(saida_json, indent=1))
  f.close()

  contador += 1


# In[82]:


data


# In[83]:


#lendo o novo json
data = json.load(open('json/dados2.json', encoding='utf8'))


# In[84]:


# Criando um grafico de Collapsible Tree em HTML
   
visualizacao = """<!DOCTYPE html>
<meta charset='utf-8'>
<style>

.node circle {
  fill: #fff;
  stroke: steelblue;
  stroke-width: 1.5px;
}

.node {
  font: 10px sans-serif;
}

.link {
  fill: none;
  stroke: #ccc;
  stroke-width: 1.5px;
}

</style>
<body>
<script src='https://d3js.org/d3.v3.min.js'></script>
<script>

root = %s;
//j = JSON.parse(valores);

var width = 1080,
    height = 2200;

var cluster = d3.layout.cluster()
    .size([height, width - 160]);

var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.y, d.x]; });

var svg = d3.select('body').append('svg')
    .attr('width', width)
    .attr('height', height)
  .append('g')
    .attr('transform', 'translate(50,0)');

console.log(root);


var nodes = cluster.nodes(root),
    links = cluster.links(nodes);

var link = svg.selectAll('.link')
    .data(links)
  .enter().append('path')
    .attr('class', 'link')
    .attr('d', diagonal);

var node = svg.selectAll('.node')
    .data(nodes)
  .enter().append('g')
    .attr('class', 'node')
    .attr('transform', function(d) { return 'translate(' + d.y + ',' + d.x + ')'; })

node.append('circle')
    .attr('r', 4.5);

node.append('text')
    .attr('dx', function(d) { return d.children ? -8 : 8; })
    .attr('dy', 3)
    .style('text-anchor', function(d) { return d.children ? 'end' : 'start'; })
    .text(function(d) { return d.name; });


d3.select(self.frameElement).style('height', height + 'px');


</script>

"""


# In[85]:


#Fechando o html
with open('html/file.html', 'w', encoding='utf8' ) as f:
    f.write(visualizacao % (valores))
    f.close()


# In[86]:


# Criando um grafico de Collapsible Tree Circular em HTML

visualizacao2 = """<!DOCTYPE html>
<meta charset="utf-8">
<style>

.node circle { fill: #fff; stroke: steelblue; stroke-width: 1.5px; }
.node { font: 11px sans-serif; }
.link { fill: none; stroke: #ccc; stroke-width: 1.5px; }

</style>
<body>
<script src="https://d3js.org/d3.v3.min.js"></script>
<script>

var diameter = 960;

var tree = d3.layout.tree()
    .size([360, diameter / 2 - 120])
    .separation(function(a, b) { return (a.parent == b.parent ? 1 : 2) / a.depth; });

var diagonal = d3.svg.diagonal.radial()
    .projection(function(d) { return [d.y, d.x / 180 * Math.PI]; });

var svg = d3.select("body").append("svg")
    .attr("width", diameter)
    .attr("height", diameter)
  .append("g")
    .attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");

root = %s;

var nodes = tree.nodes(root),
    links = tree.links(nodes);

var link = svg.selectAll(".link")
    .data(links)
  .enter().append("path")
    .attr("class", "link")
    .attr("d", diagonal);

var node = svg.selectAll(".node")
    .data(nodes)
  .enter().append("g")
    .attr("class", "node")
    .attr("transform", function(d) { return "rotate(" + (d.x - 90) + ")translate(" + d.y + ")"; })

node.append("circle")
    .attr("r", 4.5);

node.append("text")
    .attr("dy", ".31em")
    .attr("text-anchor", function(d) { return d.x < 180 ? "start" : "end"; })
    .attr("transform", function(d) { return d.x < 180 ? "translate(8)" : "rotate(180)translate(-8)"; })
    .text(function(d) { return d.name; });


d3.select(self.frameElement).style("height", diameter - 10 + "px");

</script>
"""


# In[87]:


#Salvando a visualização em HTML
with open('html/file2.html', 'w', encoding='utf8' ) as f:
    f.write(visualizacao2 % (valores))
    f.close()


# ### Trabalhando com localização

# In[88]:


#API google geocoders
g = geocoders.GoogleV3('') #Adicionar Chave da Credencial da API


# In[89]:


#Relembrando dataset
conexoes.head(10)


# In[90]:


#Verificando a localização de uma empresa exemplo
localizacao = g.geocode('Tecnifox Indústria e Comércio Ltda')
localizacao


# In[91]:


#Selecionando latitude e longitude

localizacao.latitude, localizacao.longitude


# In[92]:


#criando duas colunas novas no dataset
conexoes['Latitude'] = None
conexoes['Longitude'] = None


# In[93]:


#Buscando latitude e longitude de todas as empresas do dataset
for i in range(0, len(conexoes)):
    print(conexoes['Company'][i])
    try:
        localizacao = g.geocode(conexoes['Company'][i])
    except:
        print(conexores['Company'][i])
    if localizacao != None:
        conexoes['Latitude'][i] = localizacao.latitude
        conexoes['Longitude'][i] = localizacao.longitude


# In[94]:


#Salvando em CSV
conexoes.to_csv('datasets/conexoes_loc.csv', index=False)


# In[95]:


#Lendo o CSV Salvo
conexoes = pd.read_csv('datasets/conexoes_loc.csv')
conexoes.head()


# In[96]:


#Verificando valores máximos e minimos latitude
conexoes['Latitude'].describe()


# In[97]:


#Verificando valores máximos e minimos de longitude

conexoes['Longitude'].describe()


# In[98]:


#Selecionando menores e maiores valores
lat1, lat2 = conexoes['Latitude'].min(), conexoes['Latitude'].max()
lon1, lon2 = conexoes['Longitude'].min(), conexoes['Longitude'].max()


# In[99]:


#Criando um mapa com a localização das empresas
plt.figure(figsize=(20,15))
m = Basemap(projection='cyl', resolution='h',
            llcrnrlat = lat1, urcrnrlat = lat2,
            llcrnrlon = lon1, urcrnrlon = lon2)
m.drawcoastlines()
m.fillcontinents(color = 'palegoldenrod', lake_color = 'lightskyblue')
m.drawmapboundary(fill_color='lightskyblue')
m.scatter(conexoes['Longitude'], conexoes['Latitude'], s = 30, c = 'red', zorder = 2);


# In[100]:


#Buscando uma localização de teste pelas coordenadas geograficas
localizacao = rg.search((conexoes['Latitude'][4], conexoes['Longitude'][4]))


# In[101]:


#Veificando localização
localizacao


# In[102]:


#Selecioanando País e Cidade
localizacao[0]['cc'], localizacao[0]['name']


# In[103]:


#Criando duas novas colunas
conexoes['Cidade'] = None
conexoes['País'] = None


# In[104]:


#Buscando todos os cidades e países das empresas
for i in range(0,  len(conexoes)):
    try:
        localizacao = rg.search((conexoes['Latitude'][i], conexoes['Longitude'][i]))
        conexoes['País'][i] = localizacao[0]['cc']
        conexoes['Cidade'][i] = localizacao[0]['name']
    except:
        print(conexoes['Company'][i])


# In[105]:


#Salvando em um CSV
conexoes.to_csv('datasets/conexoes_loc_complet.csv', index=False)


# In[106]:


#Lendo o CSV
conexoes = pd.read_csv('datasets/conexoes_loc_complet.csv')
conexoes.head()


# In[107]:


#Verificando a Quantidade de contatos por país
grafico = px.histogram(x = conexoes['País'], 
                       title = 'Quantidade de Contatos por País',
                       labels={'x': 'País',})

grafico.show()


# In[108]:


#Treemapping por país, cidade e cargo 

grafico = px.treemap(conexoes[conexoes['País'].notnull()], path=['País','Cidade', 'Position', 'Full Name'] )
grafico.show()


# In[109]:


#Quantidade por cidade
grafico = px.histogram(x = conexoes['Cidade'], 
                       title = 'Quantidade de Contatos por Cidade',
                       labels={'x': 'Cidade',})
                      
grafico.show()


# In[110]:


#Treemapping por cidade

grafico = px.treemap(conexoes[conexoes['Cidade'].notnull()], path =['Cidade', 'Position', 'Full Name'])
grafico.show()


# In[111]:


#Selecionando somente empresas no Brasil

conbr = conexoes[conexoes['País'] == 'BR']

conbr.head()


# In[112]:


#Tamanho
conbr.shape


# In[113]:


#Treemapping somente BR
grafico = px.treemap(conbr, path=['Cidade', 'Position', 'Full Name'])

grafico.show()


# ## Aprendizagem Não Supervisionada - Clustering

# In[114]:


#Selecionando as latitudes e longitudes
x = conexoes.iloc[:, 7:9].dropna().values

x


# In[115]:


#Aplicando o algoritimo KMeans com 15 clusters
kmeans = KMeans(n_clusters=15)
kmeans.fit(x)


# In[116]:


#Verificando os clusters
kmeans.labels_


# In[117]:


#Verificando os centroides
kmeans.cluster_centers_


# In[118]:


#Criando arquivos KML para Google Earth
kml_contatos = simplekml.Kml()


# In[119]:


#Criando arquivos de conexoes em KML para Google Earth
for i in range(0, len(conexoes)):
    kml_contatos.newpoint(name=conexoes['Full Name'][i],
                         coords=[(conexoes['Longitude'][i],
                                 conexoes['Latitude'][i])])
kml_contatos.save('kml/conexoes.kml')


# In[120]:


#Criando arquivos dos grupos em KML para Google Earth

kml_grupos = simplekml.Kml()
for i in range(len(kmeans.cluster_centers_)):
    kml_grupos.newpoint(name = 'Grupo {}'.format(i),
                      coords = [(kmeans.cluster_centers_[i][1], kmeans.cluster_centers_[i][0])])
kml_grupos.save('kml/grupos.kml')


# In[121]:


#Após upload dos KML no site do google earth

from IPython.display import Image
Image(filename='img/googleearth.png')

