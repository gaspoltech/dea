import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
# from PIL import Image
st.set_page_config(layout='wide')
# load model 
# import joblib

# linear programming
import pulp
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

def main():
    st.title("Linear Programming Simulation")
    menu = ["DEA_Analysis","LP_simulation"]
    choice = st.sidebar.selectbox("Select Menu", menu)
    df = pd.read_excel('dataset_final.xlsx')

    if choice == "DEA_Analysis":
        st.write(df.head())
        # st.write(dflp.head())
    elif choice == "LP_simulation":
        # listpemda = df.Pemda.tolist()
        pilihsektor = st.sidebar.selectbox('Pilih Potensi Sektoral',df.Potensi.unique().tolist())
        df = df[df['Potensi'].isin([pilihsektor])]
        pemda = st.sidebar.selectbox('Pilih Pemda',df.Pemda.tolist())
        dfc = df[df['Pemda'].isin([pemda])]
        sektor = dfc['Sektor_group'].tolist()
        sektor = sektor[0]
        st.sidebar.write(f'Potensi Sektoral  : {sektor}')
        st.subheader(f'Nilai Efisiensi Sektoral: {dfc.Efisiensi.sum()*100}%')
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
        # st.table(dfm[['variable','value']])
        # st.write(dfm['value'][0])
        # filter frontier
        # dflp = df[df['Sektor_group'].isin([sektor])]
        dflp = df[df['Sektor_group']==sektor]
        # dflp = dflp[dflp['Efisiensi'].isin([1])]
        dflp = dflp[dflp['Efisiensi']==1.00]
        dflp = dflp.replace(to_replace=0,value=np.NAN)
        kolom = dfm.variable.unique().tolist()
        # g1,g2=st.beta_columns((3,2))
        # with g1:
        fig = go.Figure()
        fig.add_trace(go.Box(y=dflp[s1],name=s1))
        fig.add_trace(go.Box(y=dflp[s2],name=s2))
        fig.add_trace(go.Box(y=dflp[s3],name=s3))
        fig.add_trace(go.Box(y=dflp[s4],name=s4))
        fig.add_trace(go.Box(y=dflp[s5],name=s5))
        fig.add_trace(go.Box(y=dflp[s6],name=s6))
        fig.add_trace(go.Box(y=dflp[s7],name=s7))
        fig.add_trace(go.Box(y=dflp[s8],name=s8))
        fig.add_trace(go.Box(y=dflp[s9],name=s9))
        fig.add_trace(go.Scatter(x=kolom, y=dfm['value'],mode='lines',name=pemda))
        fig.update_layout(width=900)
        st.plotly_chart(fig)
        # with g2:
        #     fig0 = px.pie(values=dfm['value'],color =dfm['variable'],hole=0.6)
        #     st.plotly_chart(fig0)

        with st.beta_expander('Efficiency Analysis', expanded=False):
            c1,c2,c3,c4 = st.beta_columns((2,1,2,2))
            with c1:
                bv = dfc.iloc[0,2:15].tolist()
                v01= bv[0]
                v02= bv[1]
                v03= bv[2]
                v04= bv[3]
                # st.write(bv)
                st.text_input(label=s1,value=dfc[s1].sum()*100)
                st.text_input(label=s2,value=dfc[s2].sum()*100)
                st.text_input(label=s3,value=dfc[s3].sum()*100)
                st.text_input(label=s4,value=dfc[s4].sum()*100)
                st.text_input(label=s5,value=dfc[s5].sum()*100)
                st.text_input(label=s6,value=dfc[s6].sum()*100)
                st.text_input(label=s7,value=dfc[s7].sum()*100)
                st.text_input(label=s8,value=dfc[s8].sum()*100)
                st.text_input(label=s9,value=dfc[s9].sum()*100)
            with c2:
                st.empty()
            with c3:
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
            with c4:
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

        ef = pd.read_excel('ef20.xlsx')
        ef = ef.iloc[:,sektor].tolist()
        # st.write(ef[0])
        gr = pd.read_excel('gr20.xlsx')
        gr = gr.iloc[:,1].tolist()
        # st.write(gr[3])

        
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
        prob += ef[0]+ef[1]*v01+ef[2]*v02+ef[3]*v03+ef[4]*v04+v1*ef[5]+v2*ef[6]+v3*ef[7]+v4*ef[8]+v5*ef[9]+v6*ef[10]+v7*ef[11]+v8*ef[12]+v9*ef[13]

        # Solve the problem
        st.write("Generate Effective and Efficient Allocation")
        if st.button("Click Here to execute"):
            status = prob.solve()
            p1 =  pulp.value(v1)/100
            p2 =  pulp.value(v2)/100
            p3 =  pulp.value(v3)/100
            p4 =  pulp.value(v4)/100
            p5 =  pulp.value(v5)/100
            p6 =  pulp.value(v6)/100
            p7 =  pulp.value(v7)/100
            p8 =  pulp.value(v8)/100
            p9 =  pulp.value(v9)/100
            total = p1+p2+p3+p4+p5+p6+p7+p8+p9
            outls = [p1,p2,p3,p4,p5,p6,p7,p8,p9]
            # st.subheader(f'Total Alokasi: {int(total)}%')
            h1,h2 = st.beta_columns((5,3))
            
            with h1:
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(x=kolom, y=dfm['value'],name='Current Allocation'))
                fig1.add_trace(go.Bar(x=kolom, y=outls,name='Recommendation'))
                fig1.update_layout(width=700)
                st.plotly_chart(fig1)
            #     # st.subheader(f'Prediksi Nilai Efisiensi: {status*100}%')
            #     fig2 = go.Figure()
            #     fig2.add_trace(go.Indicator(
            #                     mode = "number+delta",
            #                     value = status*100,
            #                     title = {"text": "Prediksi Nilai Efisiensi:"},
            #                     delta = {'reference': dfc.Efisiensi.sum()*100, 'relative': True},
            #                     domain = {'x': [0.6, 1], 'y': [0, 1]},
            #                     ))
            #     fig2.update_layout(width=400,height=300)
            #     # fig2.update_layout(autosize=True)
            #     st.plotly_chart(fig2)
            with h2:
                # st.write(dfc.GrowthY.sum())
                b1='TK'
                b2='IPM'
                b3='PMTB'
                b4='IKK'
                b5='Sektor_group'
                b6='Efisiensi'
                # st.write(dfc[b1].sum())
                # st.write(dfc[b2].sum())
                growth = gr[0]+dfc[b1].sum()*gr[1]+dfc[b2].sum()*gr[2]+dfc[b3].sum()*gr[3]+dfc[b4].sum()*gr[4]+dfc[b5].sum()*gr[5]+dfc[b6].sum()*status
                # st.title(f'Prediksi Nilai Growth: {growth*100}%')
                fig3 = go.Figure()
                fig3.add_trace(go.Indicator(
                                mode = "number+delta",
                                value = status*100,
                                title = {"text": "Prediksi Nilai Efisiensi:"},
                                delta = {'reference': dfc.Efisiensi.sum()*100, 'relative': True},
                                domain = {'x': [0, 0.5], 'y': [0.6, 1]},
                                ))
                # fig3 = go.Figure()
                fig3.add_trace(go.Indicator(
                                mode = "number+delta",
                                value = growth,
                                title = {"text": "Prediksi Tingkat Growth:"},
                                delta = {'reference': dfc.GrowthY.sum()*100, 'relative': True},
                                domain = {'x': [0, 0.5], 'y': [0, 0.4]},
                                ))
                # fig3.update_layout(autosize=True)
                # fig3.update_layout(width=400,height=500)
                st.plotly_chart(fig3)
            



if __name__=='__main__':
    main()