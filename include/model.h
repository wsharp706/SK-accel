/**
 * @brief Linear model class declaration
 * @author Will Sharpsteen - wisharpsteen@gmail.com
 */
#include "matrix.h"
#include "customexceptions.h"
#include <vector>
#include <cmath>

#ifndef MODEL_H
#define MODEL_H

class model
{
    private:
    matrix rawdata;
    matrix preSTANDARD; // this guy is only used for standardized models.
    matrix X;
    matrix Xt;
    matrix XtX_inv;
    matrix Y;
    matrix residuals;
    matrix beta;
    unsigned int n;
    unsigned int response_col;
    std::vector< unsigned int > explanatory_cols;
    std::string transform;
    bool index_mode;

    public:
    model( );

    ~model( );

    model( const model& original );

    /**
     * @brief fit linear model to data matrix.
     * @param response_col position (or index) of response column variables
     * @param explanatory_cols initilizer list of all useful explanatory columns; ex. {1,2,4}
     * @param index_mode boolean flip to index for positions. index + 1 = position. default = false
     * @param transform standardize or none
     * @param weights vector of associated weights with each column
     * @exception dimSizeError thrown for incompaible matrix size
     * @return model
     */
    friend auto fitlm( const matrix& data, unsigned int response_col, std::vector< unsigned int > explanatory_cols, bool index_mode, std::string transform ) -> model;

    /**
     * @brief fit weighted linear model to data matrix.
     * @param response_col position (or index) of response column variables
     * @param explanatory_cols initilizer list of all useful explanatory columns; ex. {1,2,4}
     * @param index_mode boolean flip to index for positions. index + 1 = position. default = false
     * @param transform standardize or none.
     * @exception dimSizeError thrown for incompatible matrix size
     * @return model
     */
    friend auto fitwlm( const matrix& data, unsigned int response_col, std::vector< long double > weights, std::vector< unsigned int > explanatory_cols, bool index_mode, std::string transform ) -> model;

    /**
     * @brief number of explanatory variables used in model
     * @return int, count
     */
    auto nparam( ) const -> unsigned int;

    /**
     * @brief number of observations used in model
     * @return int, count
     */
    auto nobs( ) const -> unsigned int;

    /**
     * @brief returns beta-hat coefficient vector
     * @return column matrix beta-hat values
     */
    auto coef( ) const -> matrix;

    /**
     * @brief get SE of which beta parameter
     * @return long double SE
     * @param which beta param (intercept = 0)
     * @exception dimSizeError thrown if param is out of reach
     */
    auto getcoefSE( const int which ) const -> long double;

    /**
     * @brief get specific regression param
     * @param which int
     * @return long double
     * @exception dimError thrown if param is out of reach
     */
    auto getcoef( const int which ) const -> long double;

    /**
     * @brief compute hat matrix for model
     * @return matrix
     */
    auto hat( ) const -> matrix;

    /**
     * @brief compute model's predicted values for Y, or Y_hat
     * @return matrix
     */
    auto yhat( ) const -> matrix;

    /**
     * @brief return pre-computed residuals for model
     * @return matrix of residuals
     */
    auto res( ) const -> matrix;

    /**
     * @brief compute SSTO of model
     * @return long double
     */
    auto SSTO( ) const -> long double;

    /**
     * @brief compute SSE of model
     * @return long double
     */
    auto SSE( ) const -> long double;

    /**
     * @brief compute SSR of model
     * @return long double
     */
    auto SSR( ) const -> long double;

    /**
     * @brief compute MSE of model
     * @return long double
     */
    auto MSE( ) const -> long double;

    /**
     * @brief compute MSR of model
     * @return long double
     */
    auto MSR( ) const -> long double;

    /**
     * @brief compute variance-covariance matrix of assorted model elements
     * @param which "b", "e"
     * @return matrix
     */
    auto vcov( const std::string& which ) const -> matrix;

    /**
     * @brief Standard error of each parameter
     * @return column matrix of each parameters associated SE
     */
    auto SE( ) const -> matrix;

    /**
     * @brief Get fitted value Y-hat from model
     * @param preds row matrix of each X_i observation
     * @return long double
     * @exception dimSizeError thrown if preds matrix is incompatible size
     */
    auto predict( const matrix& preds ) const -> long double;

    /**
     * @brief Prediction variance s(Y_hat)
     * @param preds row matrix of observations
     * @exception dimSizeError thrown if size of preds param is not correct
     * @return long double
     */
    auto s_yh( const matrix& preds ) const -> long double;

    /**
     * @brief compute beta_hat confidence intervals for each parameter
     * @param alpha confidence level
     * @return matrix with parameter index on left col, left bound of ci in middle col, and right bound of ci in right col, for every beta_hat
     */
    auto bci( const float alpha ) const -> matrix;

    /**
     * @brief compute mean reponse of prediction confidence interval
     * @param alpha confidence level
     * @param preds row matrix of observations
     * @return matrix with left bound of CI as first element, right bound of CI as second
     * @exception dimSizeError thrown if size of preds matrix is of incompatible size
     */
    auto mrci( const float alpha, const matrix& preds ) const -> matrix;

    /**
     * @brief compute requested extra sum of squares
     * @param new_ std::vector< int > vector of integers specifying which paramaters are on the "left", or those not being given
     * @param given std::vector< int > vector of integers specifying which parameters are taken as given
     * @exception dimSizeError thrown if vector of integers has a parameter identifier not in the model
     * @return long int
     */
    auto ESSR( const std::vector< unsigned int >& new_, const std::vector< unsigned int >& given ) const -> long double;

    /**
     * @brief compute s{pred} of predictions
     * @param preds row matrix of observations
     * @return long double
     * @exception dimSizeError thrown if preds is of incompatible size
     */
    auto spred( const matrix& preds ) const -> long double;

    /**
     * @brief compute prediction interval of new observations from data
     * @param preds row matrix of observations
     * @param alpha confidence level
     * @return matrix with left bound of CI as first element, right bound of CI as second
     * @exception dimSizeError thrown if preds is of incompatible size
     */
    auto pi( const float alpha, const matrix& preds ) const -> matrix;

    /**
     * @brief computes F-test statistic of removing selected parameters from model
     * @param to_drop std::vector< int > indexes reporting which observations to drop, x1, x2, ...
     * @param pval true to return p-val, false to return test statistic
     * @return p-value/test stat long double of test
     */
    auto testF( const std::vector< unsigned int >& to_drop, bool pval = true ) const -> long double;

    /**
     * @brief compute R^2 coefficent of determination
     * @return long double
     */
    auto R2( ) const -> long double;

    /**
     * @brief compute coefficent of partial determination for given set of explanatory variables
     * @return long double
     * @param which vector of integers indicating which explanatory variable as reponse. ex, { 1, 2 } is R^2_{1,2|rest}
     */
    auto R2partial( const std::vector< unsigned int >& which ) const -> long double;

    /**
     * @brief compute coefficent of partial determination for given set of explanatory variables against others
     * @return long double
     * @param which vector of integers indicating which explanatory variable as reponse. ex, { 1, 2 } is R^2_{1,2|against}
     * @param against vector of integers indicatating which explanatory variables as against
     */
    auto R2partial( const std::vector< unsigned int >& which, const std::vector< unsigned int >& given ) const -> long double;

    /**
     * @brief compute sqrt of coefficent of partial determination for given set of explanatory variables
     * @return long double
     * @param which vector of integers indicating which explanatory variable as reponse. ex, { 1, 2 } is r_{1,2|rest}
     */
    auto rpartial( const std::vector< unsigned int >& which ) const -> long double;

    /**
     * @brief compute sqrt of coefficent of partial determination for given set of explanatory variables against others
     * @return long double
     * @param which vector of integers indicating which explanatory variable as reponse. ex, { 1, 2 } is r_{1,2|against}
     * @param against vector of integers indicatating which explanatory variables as against
     */
    auto rpartial( const std::vector< unsigned int >& which, const std::vector< unsigned int >& given ) const -> long double;

    /**
     * @brief compute XX correlation matrix
     * @return symmetric matrix of r_{i,j} coefficients
     */
    auto corXX( ) const -> matrix;

    /**
     * @brief compute YX correlation matrix
     * @return column matrix
     */
    auto corYX( ) const -> matrix;

    /**
     * @brief de-standardize beta parameters
     * @return column matrix of de-standardized beta parameters
     */
    auto destandardize( ) const -> matrix;

};

auto operator<<( std::ostream& os, model& t_model ) -> std::ostream&;

inline auto fitlm( const matrix& data, unsigned int response_col, std::vector< unsigned int > explanatory_cols, bool index_mode = false, std::string transform = "none" ) -> model
{
    if ( transform == "standardize" )
    {
        matrix dataNew;
        auto oldY = data.getcol( response_col - !index_mode );
        auto oldYm = mean( oldY );
        auto oldYs = s(oldY);
        auto newY = oldY;
        for ( int i = 0; i < oldY.size( ); ++i )
        {
            newY[ i ] = ( 1 / (sqrtl(oldY.size( )-1))) * ( (oldY[i]-oldYm) / oldYs );
        }
        dataNew.appendcol( newY );
        std::vector< long double > oldX;
        long double oldXm;
        long double oldXs;
        std::vector< long double > newX;
        for ( auto &row : explanatory_cols )
        {
            oldX = data.getcol( row - !index_mode );
            oldXm = mean(oldX);
            oldXs = s(oldX);
            for ( int i = 0; i < oldX.size( ); ++i )
            {
                newX.push_back( ( 1 / (sqrtl(oldX.size( )-1))) * ( (oldX[i]-oldXm) / oldXs ) );
            }
            dataNew.appendcol(newX);
            newX.clear( );
        }

        model result;
        result.rawdata = dataNew;
        result.response_col = response_col;
        result.explanatory_cols = explanatory_cols;
        result.n = dataNew.nrow( );
        matrix y;
        y.appendcol( dataNew.getcol( response_col - !index_mode ) );
        auto X = init( 1, dataNew.nrow( ), 1 );
        for ( auto &t_col : explanatory_cols )
        {
            X.appendcol( dataNew.getcol( t_col - !index_mode ) );
        }
        result.X = X;
        result.Y = y;
        auto Xt = X.t( );
        result.Xt = Xt;
        auto xtx = Xt % X;
        result.XtX_inv = invert( xtx, "qr" );
        result.beta = ( result.XtX_inv % Xt % y );
        result.residuals = result.Y - result.X % result.beta;
        result.transform = transform;
        result.index_mode = index_mode;
        result.preSTANDARD = data;
        return result;
    }
    else
    {
        model result;
        result.rawdata = data;
        result.response_col = response_col;
        result.explanatory_cols = explanatory_cols;
        result.n = data.nrow( );
        matrix y;
        y.appendcol( data.getcol( response_col - !index_mode ) );
        auto X = init( 1, data.nrow( ), 1 );
        for ( auto &t_col : explanatory_cols )
        {
            X.appendcol( data.getcol( t_col - !index_mode ) );
        }
        result.X = X;
        result.Y = y;
        auto Xt = X.t( );
        result.Xt = Xt;
        auto xtx = Xt % X;
        result.XtX_inv = invert( xtx, "qr" );
        result.beta = ( result.XtX_inv % Xt % y );
        result.residuals = result.Y - result.X % result.beta;
        result.transform = transform;
        result.index_mode = index_mode;
        return result;
    }
}

inline auto fitwlm( const matrix& data, unsigned int response_col, std::vector< long double > weights, std::vector< unsigned int > explanatory_cols, bool index_mode, std::string transform ) -> model
{
    matrix W = init(0, weights.size( ), weights.size( ) );
    int i = 0;
    for ( auto &elem : weights )
    {
        W.setelem( elem, i, i, true );
        ++i;
    }
    if ( transform == "standardize" )
    {
        matrix dataNew;
        auto oldY = data.getcol( response_col - !index_mode );
        auto oldYm = mean( oldY );
        auto oldYs = s(oldY);
        auto newY = oldY;
        for ( int i = 0; i < oldY.size( ); ++i )
        {
            newY[ i ] = ( 1 / (sqrtl(oldY.size( )-1))) * ( (oldY[i]-oldYm) / oldYs );
        }
        dataNew.appendcol( newY );
        std::vector< long double > oldX;
        long double oldXm;
        long double oldXs;
        std::vector< long double > newX;
        for ( auto &row : explanatory_cols )
        {
            oldX = data.getcol( row - !index_mode );
            oldXm = mean(oldX);
            oldXs = s(oldX);
            for ( int i = 0; i < oldX.size( ); ++i )
            {
                newX.push_back( ( 1 / (sqrtl(oldX.size( )-1))) * ( (oldX[i]-oldXm) / oldXs ) );
            }
            dataNew.appendcol(newX);
            newX.clear( );
        }

        model result;
        result.rawdata = dataNew;
        result.response_col = response_col;
        result.explanatory_cols = explanatory_cols;
        result.n = dataNew.nrow( );
        matrix y;
        y.appendcol( dataNew.getcol( response_col - !index_mode ) );
        auto X = init( 1, dataNew.nrow( ), 1 );
        for ( auto &t_col : explanatory_cols )
        {
            X.appendcol( dataNew.getcol( t_col - !index_mode ) );
        }
        result.X = X % W;
        result.Y = y;
        auto Xt = X.t( );
        result.XtX_inv = invert( Xt % X, "qr" );
        result.Xt = Xt;
        result.beta = ( result.XtX_inv % Xt % y );
        result.residuals = result.Y - result.X % result.beta;
        result.transform = transform;
        result.index_mode = index_mode;
        result.preSTANDARD = data;
        return result;
    }
    else
    {
        model result;
        result.rawdata = data;
        result.response_col = response_col;
        result.explanatory_cols = explanatory_cols;
        result.n = data.nrow( );
        matrix y;
        y.appendcol( data.getcol( response_col - !index_mode ) );
        auto X = init( 1, data.nrow( ), 1 );
        for ( auto &t_col : explanatory_cols )
        {
            X.appendcol( data.getcol( t_col - !index_mode ) );
        }
        result.X = X;
        result.Y = y;
        auto Xt = X.t( );
        result.Xt = Xt;
        auto xtx = Xt % X;
        result.XtX_inv = invert( xtx, "qr" );
        result.beta = ( result.XtX_inv % Xt % y );
        result.residuals = result.Y - result.X % result.beta;
        result.transform = transform;
        result.index_mode = index_mode;
        return result;
    }
}

#endif