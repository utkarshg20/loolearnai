holdings='''
            <style>
            .holding{
                float: center;
                font-weight: 600;
                font-size: 35px;
                font-family: arial;
            }
            </style>
            <body>
            <center><p1 class='holding'> Optimized Portfolio Holdings </p1></center>
            </body>
            '''

            st.markdown(holdings, unsafe_allow_html=True)
            st.plotly_chart(fig_pie)
        with stats:
            st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
            st.write('___________')
            st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
            st.write('___________')
            st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
            st.write('___________')
            st.subheader('''Discrete allocation:
            {}'''.format(allocation))
            st.write('___________')
            st.subheader("Funds remaining: ${:.2f}".format(leftover))
        st.write('___________________________')
        col1, col2=st.columns(2)
        with col1:
            st.subheader("Optimized Max Sharpe Portfolio Performance")
            st.image(fig_efficient_frontier)
        with col2:
            st.subheader("Correlation between stocks")
            st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
        col1,col2=st.columns(2)
        with col1:
            st.subheader('Price of Individual Stocks')
            st.plotly_chart(fig_price)
        with col2:
            st.subheader('Cumulative Returns of Stocks Starting with $100')
            st.plotly_chart(fig_cum_returns)


