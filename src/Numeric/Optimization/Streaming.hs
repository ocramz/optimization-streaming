module Numeric.Optimization.Streaming where

-- import Streaming
import qualified Streaming.Prelude as S

-- import Control.Applicative

import Data.Foldable

import qualified Data.Vector as V



linearLoss :: Num a => V.Vector a -> V.Vector a -> a
linearLoss w x = V.sum $ V.zipWith (*) w x





{- from `vlearn`

-- | The input space.
type X = Vector Float
-- | The output space.
type Y = Float
-- | The parameter space.
type D = Vector Float
-- | The prediction loss function.
type Loss = D -> X -> Y -> D
-- | The prediction loss function (derivative).
type DLoss = D -> X -> Y -> D

-- |Linear Model
type Hypothesis = D -> X -> Y
hypLinear :: Hypothesis
hypLinear w x = V.sum $ V.zipWith (*) w x

-}
