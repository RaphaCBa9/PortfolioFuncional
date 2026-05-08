module Finance
  ( covMatrix
  , precomputeRetsV
  , precomputeMeanRets
  , portfolioReturn
  , portfolioVolatility
  , sharpeRatio
  , evaluatePortfolio
  ) where

import qualified Data.Vector         as V
import qualified Data.Vector.Unboxed as VU
import           Types

tradingDays :: Double
tradingDays = 252.0

mean :: VU.Vector Double -> Double
mean v = VU.sum v / fromIntegral (VU.length v)

covariance :: VU.Vector Double -> VU.Vector Double -> Double
covariance xs ys =
  let n  = fromIntegral (VU.length xs)
      mx = mean xs
      my = mean ys
  in VU.sum (VU.zipWith (\x y -> (x - mx) * (y - my)) xs ys) / n

covMatrix :: ReturnMatrix -> VU.Vector Double
covMatrix rets =
  let n   = length rets
      arr = V.fromList rets
  in VU.fromList [ covariance (arr V.! i) (arr V.! j)
                 | i <- [0 .. n-1], j <- [0 .. n-1] ]

precomputeRetsV :: ReturnMatrix -> V.Vector DailyReturns
precomputeRetsV = V.fromList

precomputeMeanRets :: ReturnMatrix -> VU.Vector Double
precomputeMeanRets = VU.fromList . map mean

portfolioReturn :: VU.Vector Double -> WeightVec -> Double
portfolioReturn meanRets w = VU.sum (VU.zipWith (*) w meanRets) * tradingDays


portfolioVolatility :: VU.Vector Double -> WeightVec -> Double
portfolioVolatility cov w =
  let n  = VU.length w
      cw = VU.generate n $ \i ->
             VU.foldl' (\acc j -> acc + (cov VU.! (i*n+j)) * (w VU.! j))
                       0.0
                       (VU.enumFromN 0 n)
      variance = VU.sum (VU.zipWith (*) w cw)
  in sqrt (max 0.0 variance) * sqrt tradingDays

sharpeRatio :: Double -> Double -> Double
sharpeRatio ret vol
  | vol <= 0  = 0
  | otherwise = ret / vol

evaluatePortfolio :: [Ticker] -> VU.Vector Double -> VU.Vector Double
                  -> WeightVec -> PortfolioResult
evaluatePortfolio tickers meanRets cov w =
  let r  = portfolioReturn  meanRets w
      v  = portfolioVolatility cov w
      sr = sharpeRatio r v
  in PortfolioResult tickers w r v sr
