module Numeric.Optimization.Streaming.Algorithms where



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



-- |Learning rates. 
data LearningRate  = InverseLearningRate !Float
                   | ConstantLearningRate !Float deriving (Eq, Show)
λ :: LearningRate -> Int -> Float
λ (InverseLearningRate c) t = c / fromIntegral t
λ (ConstantLearningRate c) _ = c

-- | Algorithms.
data  AlgoState = SGDParams !LearningRate !D !Int
                | NAGParams !LearningRate !D !Int !D !D !Float deriving (Eq, Show, Generic)
instance NFData AlgoState

modelParam :: AlgoState -> D
modelParam (SGDParams _ x _)       = x
modelParam (NAGParams _ x _ _ _ _) = x
descend :: DLoss -> X -> Y -> AlgoState -> AlgoState
-- | Stochastic Gradient Descent
descend (δlc) x y (SGDParams lr ω t) = SGDParams lr ω' (t+1)
  where grad = δlc ω x y
        ω'       =  V.zipWith (\o g-> o - g * λ lr t) ω grad
-- | Normalized Adaptive Gradient
descend (δlc) x y (NAGParams lr ω t s gG nN) = NAGParams lr ω'' (t+1) s' gG' nN'
  where ω'                    = V.zipWith3 updateω ω s x
          where updateω ωi si xi = if abs xi > si then ωi*si/xi else ωi
        s'                    = V.zipWith updatescale s x
          where updatescale si xi = max xia si where xia = abs xi
        nN'                   = nN + V.sum ( V.zipWith sumsqratio x s')
          where sumsqratio xi si = if si==0 then 0 else xi*xi/(si*si)
        gG'                   = V.zipWith updateG gG grad
          where updateG gGi gi = gGi + gi * gi
        ω''                   =  V.zipWith4 updateweight ω' grad s' gG'
          where
            updateweight ωi gi si gGi = if si==0
                                           then ωi
                                           else ωi - λ lr t * sqrt tf * gi / (sqrt ( gGi * nN') * si)
            tf   = fromIntegral t
        grad = δlc ω' x y

-- | Helpers: base initial models for all the algorithms.
initialSGDModel :: Int -> Float -> AlgoState
initialSGDModel d c = SGDParams (InverseLearningRate c) ω0 1
    where
      ω0 = fromList $ Prelude.replicate d 0

initialNAGModel :: Int -> Float -> AlgoState
initialNAGModel d c = NAGParams (ConstantLearningRate c) ω0 1 ω0 ω0 0
    where
ω0 = fromList $ Prelude.replicate d 0


-}
