{-# LANGUAGE OverloadedStrings #-}
module DataLoader
  ( dowTickers
  , loadAllReturns
  , loadAllReturnsQ1
  ) where

import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv             as Csv
import qualified Data.Map.Strict      as Map
import           Data.Set             (Set)
import qualified Data.Set             as Set
import qualified Data.Vector          as V
import qualified Data.Vector.Unboxed  as VU
import           System.IO            (hPutStrLn, stderr)
import           Types

-- | 30 components of the DJIA (~2025 composition)
dowTickers :: [Ticker]
dowTickers =
  [ "AAPL", "AMGN", "AXP",  "BA",   "CAT"
  , "CRM",  "CSCO", "CVX",  "DIS",  "DOW"
  , "GS",   "HD",   "HON",  "IBM",  "JNJ"
  , "JPM",  "KO",   "MCD",  "MMM",  "MRK"
  , "MSFT", "NKE",  "NVDA", "PG",   "SHW"
  , "TRV",  "UNH",  "V",    "VZ",   "WMT"
  ]

-- ---------------------------------------------------------------------------
-- CSV row: Date, Close

data PriceRow = PriceRow
  { prDate  :: !String
  , prClose :: !Double
  }

instance Csv.FromNamedRecord PriceRow where
  parseNamedRecord r = PriceRow
    <$> r Csv..: "Date"
    <*> r Csv..: "Close"

-- | Load a ticker's prices from a local CSV file, return (date -> price) map.
loadPriceMap :: FilePath -> IO (Maybe (Map.Map String Double))
loadPriceMap path = do
  bs <- BL.readFile path
  case Csv.decodeByName bs of
    Left err -> do
      hPutStrLn stderr $ "  CSV parse error [" ++ path ++ "]: " ++ err
      return Nothing
    Right (_, rows) ->
      let m = Map.fromList [ (prDate r, prClose r)
                           | r <- V.toList rows
                           , prClose r > 0 ]
      in return (if Map.null m then Nothing else Just m)

-- | Compute simple daily returns from a sorted price map.
toReturnMap :: Map.Map String Double -> Map.Map String Double
toReturnMap prices =
  let sorted = Map.toAscList prices
  in Map.fromList
       [ (d2, (p2 - p1) / p1)
       | ((_, p1), (d2, p2)) <- zip sorted (tail sorted) ]

-- | Align return maps to their common dates and build the return matrix.
alignMaps :: [(Ticker, Map.Map String Double)]
          -> ([Ticker], ReturnMatrix)
alignMaps pairs =
  let common :: Set String
      common = foldr1 Set.intersection (map (Map.keysSet . snd) pairs)
      dates  = Set.toAscList common
      extract m = VU.fromList [ m Map.! d | d <- dates ]
  in (map fst pairs, map (extract . snd) pairs)

-- ---------------------------------------------------------------------------

loadReturns :: FilePath -> IO ([Ticker], ReturnMatrix)
loadReturns dir = do
  results <- mapM (loadOne dir) dowTickers
  let valid = [ (t, m) | (t, Just m) <- zip dowTickers results ]
  if null valid
    then error $ "No data loaded from " ++ dir ++ ". Run fetch_data.py first."
    else do
      let (tickers, matrix) = alignMaps valid
      hPutStrLn stderr $ "  Loaded: " ++ show (length tickers)
                       ++ " ativos, "
                       ++ show (VU.length (head matrix)) ++ " pregões."
      return (tickers, matrix)
  where
    loadOne d ticker = do
      let path = d ++ "/" ++ ticker ++ ".csv"
      mPrices <- loadPriceMap path
      return (fmap toReturnMap mPrices)

-- | Load H2 2025 data (training period)
loadAllReturns :: IO ([Ticker], ReturnMatrix)
loadAllReturns = loadReturns "data/raw/h2_2025"

-- | Load Q1 2025 data (out-of-sample validation)
loadAllReturnsQ1 :: IO ([Ticker], ReturnMatrix)
loadAllReturnsQ1 = loadReturns "data/raw/q1_2025"
