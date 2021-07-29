import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
# from PIL import Image

# load model 
# import joblib

# linear programming
import pulp
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

def main():
    st.title("DEA and Linear Programming Simulation")
    menu = ["DEA","LP_simulation"]
    choice = st.sidebar.selectbox("Select Menu", menu)
    df = pd.read_excel('deasample.xlsx')
    # dflp = pd.read_excel('lpmodel.xlsx')

    if choice == "DEA":
        st.write(df.head())
        # st.write(dflp.head())
    elif choice == "LP_simulation":
        # listpemda = df.Pemda.tolist()
        pilihsektor = st.sidebar.selectbox('Pilih Sektor',df.Sektor_group.unique().tolist())
        df = df[df['Sektor_group'].isin([pilihsektor])]
        pemda = st.sidebar.selectbox('Pilih Pemda',df.Pemda.tolist())
        dfc = df[df['Pemda'].isin([pemda])]
        sektor = dfc['Sektor_group'].tolist()
        sektor = sektor[0]
        st.sidebar.write(f'Sektor : {sektor}')
        st.subheader(f'Nilai Efisiensi base: {dfc.Efisiensi.sum()*100}%')
        st.subheader(f'Nilai Growth base: {dfc.GrowthY.sum()*100}%')
        s1='PelayananUmum'
        s2='Pendidikan'
        s3='PerlindunganSosial'
        s4='KetertibandanKeamanan'
        s5='Ekonomi'
        s6='LingkunganHidup'
        s7='PerumahandanFasilitasUmum'
        s8='Kesehatan'
        s9='PariwisatadanBudaya'
        
        dfm = pd.melt(dfc,id_vars=['Pemda'],value_vars=[s1,s2,s3,s4,s5,s6,s7,s8,s9])
        st.table(dfm[['variable','value']])
        st.write(dfm['value'][0])
        
        dflp = df[df['Sektor_group'].isin([sektor])]
        dflp = dflp[dflp['Efisiensi'].isin([1])]
        dflp = dflp.replace(to_replace=0,value=np.NAN)

        c1,c2 = st.beta_columns((1,1))
        with c1:
            #min value
            v1min = st.number_input(label="PelayananUmum min",value=dflp['PelayananUmum'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v2min = st.number_input(label="Pendidikan min",value=dflp['Pendidikan'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v3min = st.number_input(label="PerlindunganSosial min",value=dflp['PerlindunganSosial'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v4min = st.number_input(label="KetertibandanKeamanan min",value=dflp['KetertibandanKeamanan'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v5min = st.number_input(label="Ekonomi min",value=dflp['Ekonomi'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v6min = st.number_input(label="LingkunganHidup min",value=dflp['LingkunganHidup'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v7min = st.number_input(label="PerumahandanFasilitasUmum min",value=dflp['PerumahandanFasilitasUmum'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v8min = st.number_input(label="Kesehatan min",value=dflp['Kesehatan'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v9min = st.number_input(label="PariwisatadanBudaya min",value=dflp['PariwisatadanBudaya'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
        with c2:
            #max value
            v1max = st.number_input(label="PelayananUmum max",value=dflp['PelayananUmum'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v2max = st.number_input(label="Pendidikan max",value=dflp['Pendidikan'].max()*100,min_value=0.0, max_value=100.0, step=1.0)
            v3max = st.number_input(label="PerlindunganSosial max",value=dflp['PerlindunganSosial'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v4max = st.number_input(label="KetertibandanKeamanan max",value=dflp['KetertibandanKeamanan'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v5max = st.number_input(label="Ekonomi max",value=dflp['Ekonomi'].max()*100.0, max_value=100.0, step=1.0)
            v6max = st.number_input(label="LingkunganHidup max",value=dflp['LingkunganHidup'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v7max = st.number_input(label="PerumahandanFasilitasUmum max",value=dflp['PerumahandanFasilitasUmum'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v8max = st.number_input(label="Kesehatan max",value=dflp['Kesehatan'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            v9max = st.number_input(label="PariwisatadanBudaya max",value=dflp['PariwisatadanBudaya'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
        
        kolom = dfm.variable.unique().tolist()
        fig = go.Figure()
        fig.add_trace(go.Box(y=dflp['PelayananUmum'],name=s1))
        fig.add_trace(go.Box(y=dflp['Pendidikan'],name=s2))
        fig.add_trace(go.Box(y=dflp['PerlindunganSosial'],name=s3))
        fig.add_trace(go.Box(y=dflp['KetertibandanKeamanan'],name=s4))
        fig.add_trace(go.Box(y=dflp['Ekonomi'],name=s5))
        fig.add_trace(go.Box(y=dflp['LingkunganHidup'],name=s6))
        fig.add_trace(go.Box(y=dflp['PerumahandanFasilitasUmum'],name=s7))
        fig.add_trace(go.Box(y=dflp['Kesehatan'],name=s8))
        fig.add_trace(go.Box(y=dflp['PariwisatadanBudaya'],name=s9))
        fig.add_trace(go.Scatter(x=kolom, y=dfm['value'],mode='lines',name='lines'))

        st.plotly_chart(fig)


        b=1.14858376491401
        b1=-0.271060285793268
        b2=-0.180869366294671
        b3=0
        b4=-2.24024489915018
        b5=-0.264895207035489
        b6=-0.598522665494524
        b7=-0.808410732227263
        b8=-0.695363737389888
        b9=-12.6087311293848

        g=0.158595264296244
        g1=-0.155249447354189/100
        g2=-0.202504523896746/100
        g3=0/100
        g4=-0.17635372043293/100
        g5=-0.110917428788896/100
        g6=-0.480869546347052/100
        g7=-0.115334459924879/100
        g8=-0.168888225204282/100
        g9=-0.715529411210068/100

        # g=0.158595264296244
        # g1=-0.155249447354189
        # g2=-0.202504523896746
        # g3=0
        # g4=-0.17635372043293
        # g5=-0.110917428788896
        # g6=-0.480869546347052
        # g7=-0.115334459924879
        # g8=-0.168888225204282
        # g9=-0.715529411210068

        # b=g
        # b1=g1
        # b2=g2
        # b3=g3
        # b4=g4
        # b5=g5
        # b6=g6
        # b7=g7
        # b8=g8
        # b9=g9

        # Create the model
        prob = LpProblem(name="Allocation Optimization",sense=LpMaximize)
        # Initialize the decision variables
        v1 = LpVariable(name="PelayananUmum", lowBound=0, cat="Float")
        v2 = LpVariable(name="Pendidikan", lowBound=0, cat="Float")
        v3 = LpVariable(name="PerlindunganSosial", lowBound=0, cat="Float")
        v4 = LpVariable(name="KetertibandanKeamanan", lowBound=0, cat="Float")
        v5 = LpVariable(name="Ekonomi", lowBound=0, cat="Float")
        v6 = LpVariable(name="LingkunganHidup", lowBound=0, cat="Float")
        v7 = LpVariable(name="PerumahandanFasilitasUmum", lowBound=0, cat="Float")
        v8 = LpVariable(name="Kesehatan", lowBound=0, cat="Float")
        v9 = LpVariable(name="PariwisatadanBudaya", lowBound=0, cat="Float")
        # Add the constraints to the model
        prob += (v1+v2+v3+v4+v5+v6+v7+v8+v9 == 100, "full_constraint")
        prob += (v1 >= v1min, "v1min")
        prob += (v2 >= v2min, "v2min")
        prob += (v3 >= v3min, "v3min")
        prob += (v4 >= v4min, "v4min")
        prob += (v5 >= v5min, "v5min")
        prob += (v6 >= v6min, "v6min")
        prob += (v7 >= v7min, "v7min")
        prob += (v8 >= v8min, "v8min")
        prob += (v9 >= v9min, "v9min")
        prob += (v1 <= v1max, "v1max")
        prob += (v2 <= v2max, "v2max")
        prob += (v3 <= v3max, "v3max")
        prob += (v4 <= v4max, "v4max")
        prob += (v5 <= v5max, "v5max")
        prob += (v6 <= v6max, "v6max")
        prob += (v7 <= v7max, "v7max")
        prob += (v8 <= v8max, "v8max")
        prob += (v9 <= v9max, "v9max")
        #Objective
        prob += b+v1*b1+v2*b2+v3*b3+v4*b4+v5*b5+v6*b6+v7*b7+v8*b8+v9*b9

        # Solve the problem
        st.write("Effective and Efficient Allocation")
        if st.button("Click Here to Solve"):
            status = prob.solve()
            st.write(f"PelayananUmum: {float(pulp.value(v1))}%")
            st.write(f"Pendidikan: {float(pulp.value(v2))}%")
            st.write(f"PerlindunganSosial: {float(pulp.value(v3))}%")
            st.write(f"KetertibandanKeamanan: {float(pulp.value(v4))}%")
            st.write(f"Pendidikan: {float(pulp.value(v5))}%")
            st.write(f"LingkunganHidup: {float(pulp.value(v6))}%")
            st.write(f"PerumahandanFasilitasUmum: {float(pulp.value(v7))}%")
            st.write(f"Kesehatan: {float(pulp.value(v8))}%")
            st.write(f"PariwisatadanBudaya: {float(pulp.value(v9))}%")
            p1 =  pulp.value(v1)
            p2 =  pulp.value(v2)
            p3 =  pulp.value(v3)
            p4 =  pulp.value(v4)
            p5 =  pulp.value(v5)
            p6 =  pulp.value(v6)
            p7 =  pulp.value(v7)
            p8 =  pulp.value(v8)
            p9 =  pulp.value(v9)
            total = p1+p2+p3+p4+p5+p6+p7+p8+p9
            st.title(f'Total Alokasi: {int(total)}%')
            st.title(f'Prediksi Nilai Efisiensi: {status*100}%')

            growth = g+p1*g1+p2*g2+p3*g3+p4*g4+p5*g5+p6*g6+p7*g7+p8*g8+p9*g9

            st.title(f'Prediksi Nilai Growth: {growth*100}%')



if __name__=='__main__':
    main()