module Types
  ( Ticker
  , DailyReturns
  , WeightVec
  , ReturnMatrix
  , PortfolioResult (..)
  , showResult
  ) where

import qualified Data.Vector.Unboxed as VU
import           Data.List           (intercalate)

type Ticker      = String
type DailyReturns = VU.Vector Double
type WeightVec   = VU.Vector Double
-- | One DailyReturns vector per asset, all same length and date-aligned
type ReturnMatrix = [DailyReturns]

data PortfolioResult = PortfolioResult
  { prTickers    :: ![Ticker]
  , prWeights    :: !WeightVec
  , prReturn     :: !Double
  , prVolatility :: !Double
  , prSharpe     :: !Double
  } deriving (Eq)

instance Ord PortfolioResult where
  compare a b = compare (prSharpe a) (prSharpe b)

showResult :: PortfolioResult -> String
showResult pr = unlines
  [ "Ativos:                 " ++ intercalate ", " (prTickers pr)
  , "Pesos:                  " ++ unwords (map fmt $ VU.toList (prWeights pr))
  , "Retorno anualizado:     " ++ fmt2 (prReturn pr * 100) ++ "%"
  , "Volatilidade anualizada:" ++ fmt2 (prVolatility pr * 100) ++ "%"
  , "Sharpe Ratio:           " ++ fmt4 (prSharpe pr)
  ]
  where
    fmt  x = show (fromIntegral (round (x * 100) :: Int) / 100.0 :: Double)
    fmt2 x = show (fromIntegral (round (x * 100) :: Int) / 100.0 :: Double)
    fmt4 x = show (fromIntegral (round (x * 10000) :: Int) / 10000.0 :: Double)
