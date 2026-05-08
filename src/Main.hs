module Main (main) where

import qualified Data.Vector         as V
import qualified Data.Vector.Unboxed as VU
import           Control.DeepSeq     (force)
import           Control.Exception   (evaluate)
import           Data.Time.Clock     (diffUTCTime, getCurrentTime)
import           System.Environment  (getArgs)
import           System.IO           (hPutStrLn, stderr)

import           DataLoader
import           Finance             (covMatrix, precomputeMeanRets, portfolioReturn, portfolioVolatility, sharpeRatio)
import           Simulation          (combinations, runSimulation, simulateCombination)
import           Types

usageMsg :: String
usageMsg = unlines
  [ "carteira — otimização de portfólio Dow Jones"
  , "Flags opcionais:"
  , "  --max-combos N   testa apenas N combinações (padrão: todas ~30M)"
  , "  --q1             valida a melhor carteira no Q1 2025 (out-of-sample)"
  , "  --benchmark      compara sequencial vs paralelo (5 execuções cada)"
  ]

parseArgs :: [String] -> (Maybe Int, Bool, Bool)
parseArgs []                      = (Nothing, False, False)
parseArgs ("--max-combos":n:rest) = let (_, q1, bm) = parseArgs rest
                                    in  (Just (read n), q1, bm)
parseArgs ("--q1":rest)           = let (mc, _, bm) = parseArgs rest
                                    in  (mc, True, bm)
parseArgs ("--benchmark":rest)    = let (mc, q1, _) = parseArgs rest
                                    in  (mc, q1, True)
parseArgs (_:rest)                = parseArgs rest

main :: IO ()
main = do
  args <- getArgs
  let (maxCombos, doQ1, doBench) = parseArgs args
  putStr usageMsg

  --  Load H2 2025 data 
  hPutStrLn stderr "\n[1/3] Carregando dados H2 2025..."
  (tickers, retMatrix) <- loadAllReturns
  hPutStrLn stderr $ "     " ++ show (length tickers) ++ " ativos, "
                   ++ show (VU.length (head retMatrix)) ++ " pregões."

  --  Parallel simulation 
  hPutStrLn stderr "[2/3] Simulando carteiras em paralelo..."
  let infoStr = maybe "todas as combinações" (\n -> show n ++ " combinações") maxCombos
  hPutStrLn stderr $ "     (" ++ infoStr ++ ", 1.000.000 pesos cada)"
  t0   <- getCurrentTime
  best <- evaluate (force (runSimulation tickers retMatrix maxCombos))
  t1   <- getCurrentTime
  let elapsed = realToFrac (diffUTCTime t1 t0) :: Double

  putStrLn "\n=== Melhor carteira encontrada ==="
  putStr (showResult best)
  putStrLn $ "Tempo total: " ++ formatTime elapsed

  --  Optional benchmark ─
  if doBench
    then benchmarkRun tickers retMatrix maxCombos
    else return ()

  --  Optional Q1 validation ─
  if doQ1
    then validateQ1 best
    else return ()

-- ---------------------------------------------------------------------------

validateQ1 :: PortfolioResult -> IO ()
validateQ1 best = do
  hPutStrLn stderr "\n[3/3] Carregando dados Q1 2025 (out-of-sample)..."
  (q1Tickers, q1Matrix) <- loadAllReturnsQ1
  -- Build a map so we can look up each ticker's returns in Q1 data
  let q1Map   = zip q1Tickers q1Matrix
      bestT   = prTickers best
      -- Preserve the same ticker order as the weights
      mSubRets = sequence [ lookup t q1Map | t <- bestT ]
  case mSubRets of
    Nothing   -> putStrLn "Alguns ativos não disponíveis no Q1 — pulando validação."
    Just subRets -> do
      let meanRets = precomputeMeanRets subRets
          cov      = covMatrix subRets
          r        = portfolioReturn  meanRets (prWeights best)
          v        = portfolioVolatility cov (prWeights best)
          sr    = sharpeRatio r v
      putStrLn "\n=== Out-of-sample: Q1 2025 ==="
      putStrLn $ "Retorno anualizado:      " ++ show (r * 100) ++ "%"
      putStrLn $ "Volatilidade anualizada: " ++ show (v * 100) ++ "%"
      putStrLn $ "Sharpe Ratio:            " ++ show sr

benchmarkRun :: [Ticker] -> ReturnMatrix -> Maybe Int -> IO ()
benchmarkRun tickers retMatrix maxCombos = do
  putStrLn "\n=== Benchmark (5 execuções) ==="
  parTimes <- mapM (\_ -> timed (runSimulation    tickers retMatrix maxCombos)) [1..5 :: Int]
  seqTimes <- mapM (\_ -> timed (runSimulationSeq tickers retMatrix maxCombos)) [1..5 :: Int]
  putStrLn $ "Paralelo   (média): " ++ formatTime (avg parTimes)
  putStrLn $ "Sequencial (média): " ++ formatTime (avg seqTimes)
  putStrLn $ "Speedup:            " ++ show (avg seqTimes / avg parTimes) ++ "x"
  where
    avg xs = sum xs / fromIntegral (length xs)

timed :: PortfolioResult -> IO Double
timed result = do
  t0 <- getCurrentTime
  _ <- evaluate (force result)
  t1 <- getCurrentTime
  return (realToFrac (diffUTCTime t1 t0))

-- | Sequential simulation (for benchmark comparison)
runSimulationSeq :: [Ticker] -> ReturnMatrix -> Maybe Int -> PortfolioResult
runSimulationSeq tickers retMatrix maxCombos =
  let fullRets    = V.fromList retMatrix
      fullTickers = V.fromList tickers
      n           = length tickers
      allCombos   = combinations 20 [0 .. n-1]
      limited     = maybe allCombos (`take` allCombos) maxCombos
  in maximum (map (simulateCombination fullRets fullTickers) limited)

formatTime :: Double -> String
formatTime s
  | s < 60    = show (round s :: Int) ++ "s"
  | s < 3600  = let m = floor (s / 60) :: Int
                    sec = round (s - fromIntegral m * 60) :: Int
                in show m ++ "min " ++ show sec ++ "s"
  | otherwise = let h = floor (s / 3600) :: Int
                    m = round ((s - fromIntegral h * 3600) / 60) :: Int
                in show h ++ "h " ++ show m ++ "min"
