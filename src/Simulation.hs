module Simulation
  ( combinations
  , runSimulation
  , simulateCombination
  ) where

import           Control.DeepSeq             (NFData (..), force)
import           Control.Monad.ST            (runST, ST)
import           Control.Parallel.Strategies (parListChunk, rdeepseq, using)
import qualified Data.Vector                 as V
import qualified Data.Vector.Unboxed         as VU
import           Data.Word                   (Word32)
import qualified System.Random.MWC           as MWC
import           Finance
import           Types

instance NFData PortfolioResult where
  rnf pr = rnf (prTickers pr)
        `seq` VU.foldl' (\_ x -> x `seq` ()) () (prWeights pr)
        `seq` rnf (prReturn pr)
        `seq` rnf (prVolatility pr)
        `seq` rnf (prSharpe pr)

-- | All C(n,k) combinations from a list
combinations :: Int -> [a] -> [[a]]
combinations 0 _      = [[]]
combinations _ []     = []
combinations k (x:xs) = map (x:) (combinations (k-1) xs) ++ combinations k xs

-- | Sample a valid weight vector via Dirichlet(1,...,1) + rejection.
-- Returns Nothing when max(w) > 0.2 so the caller can retry.
sampleWeights :: MWC.Gen s -> Int -> ST s (Maybe WeightVec)
sampleWeights gen n = do
  us <- VU.replicateM n (MWC.uniform gen)
  let exps  = VU.map (\u -> negate (log (max u 1e-300))) us
      total = VU.sum exps
      ws    = VU.map (/ total) exps
  if VU.all (<= 0.2) ws
    then return (Just ws)
    else return Nothing

-- | Pure simulation for one combination of 20 asset indices.
-- MWC is seeded deterministically from the indices, making the function
-- referentially transparent (same input => same output).
simulateCombination :: V.Vector DailyReturns  -- ^ full return matrix (30 stocks)
                    -> V.Vector Ticker         -- ^ full ticker list
                    -> [Int]                   -- ^ indices of the 20 selected stocks
                    -> PortfolioResult
simulateCombination fullRets fullTickers indices = runST $ do
  let subRets    = [ fullRets V.! i | i <- indices ]
      subTickers = [ fullTickers V.! i | i <- indices ]
      n          = length indices
      meanRets   = precomputeMeanRets subRets  -- O(nDays×n), done once
      cov        = covMatrix subRets           -- O(nDays×n²), done once
      seed       = VU.fromList (map (fromIntegral :: Int -> Word32) indices)
  gen  <- MWC.initialize seed
  best <- loop gen n meanRets subTickers cov simulationsPerCombination Nothing
  let equalW = VU.replicate n (1.0 / fromIntegral n)
  return $ case best of
    Nothing -> PortfolioResult subTickers equalW
                 (portfolioReturn meanRets equalW)
                 (portfolioVolatility cov equalW)
                 0.0
    Just r  -> r
  where
    loop _ _ _ _ _ 0 acc = return acc
    loop gen n meanRets tickers cov remaining acc = do
      mw <- sampleWeights gen n
      case mw of
        Nothing -> loop gen n meanRets tickers cov (remaining - 1) acc
        Just w  ->
          let result = evaluatePortfolio tickers meanRets cov w
              acc'   = case acc of
                         Nothing   -> Just result
                         Just prev ->
                           if prSharpe result > prSharpe prev
                             then Just result
                             else Just prev
          in acc' `seq` loop gen n meanRets tickers cov (remaining - 1) acc'

simulationsPerCombination :: Int
simulationsPerCombination = 1000000

-- ---------------------------------------------------------------------------

-- | Run the full simulation in parallel across combinations.
-- Uses parListChunk with ~1000 chunks for load balancing.
runSimulation :: [Ticker]
              -> ReturnMatrix
              -> Maybe Int        -- ^ optional limit on number of combinations
              -> PortfolioResult
runSimulation tickers retMatrix maxCombos =
  let fullRets    = V.fromList retMatrix
      fullTickers = V.fromList tickers
      n           = length tickers
      allCombos   = combinations 20 [0 .. n-1]
      limited     = maybe allCombos (`take` allCombos) maxCombos
      chunkSize   = max 1 (length limited `div` 1000)
      results     = map (simulateCombination fullRets fullTickers) limited
                      `using` parListChunk chunkSize rdeepseq
  in maximum (force results)
