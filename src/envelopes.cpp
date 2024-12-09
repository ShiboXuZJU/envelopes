#include "envelopes.h"
#include <stdexcept>

// Implementation of WaveCache
std::tuple<double, double, SP<DICT<EnvFeature, REAL_VEC_SP>>>
WaveCache::getRealCache(std::string envName, double start)
{
    LLINT startOverAtol = ROUND(start / TIME_ATOL);
    LLINT baseStart = startOverAtol % this->resolutionOverAtol_;
    double shiftT = startOverAtol / this->resolutionOverAtol_ * this->resolution_;
    if (this->realCache.count(envName) == 0)
    {
        this->realCache[envName] = DICT<LLINT, SP<DICT<EnvFeature, REAL_VEC_SP>>>();
        this->realCache[envName][baseStart] = std::make_shared<DICT<EnvFeature, REAL_VEC_SP>>();
    }
    else if (this->realCache[envName].count(baseStart) == 0)
    {
        this->realCache[envName][baseStart] = std::make_shared<DICT<EnvFeature, REAL_VEC_SP>>();
    }
    return std::make_tuple(baseStart * TIME_ATOL, shiftT, this->realCache[envName][baseStart]);
}

std::tuple<double, double, SP<DICT<EnvFeature, COMPLEX_VEC_SP>>>
WaveCache::getComplexCache(std::string envName, double start)
{
    LLINT startOverAtol = ROUND(start / TIME_ATOL);
    LLINT baseStart = startOverAtol % this->resolutionOverAtol_;
    double shiftT = startOverAtol / this->resolutionOverAtol_ * this->resolution_;
    if (this->complexCache.count(envName) == 0)
    {
        this->complexCache[envName] = DICT<LLINT, SP<DICT<EnvFeature, COMPLEX_VEC_SP>>>();
        this->complexCache[envName][baseStart] = std::make_shared<DICT<EnvFeature, COMPLEX_VEC_SP>>();
    }
    else if (this->complexCache[envName].count(baseStart) == 0)
    {
        this->complexCache[envName][baseStart] = std::make_shared<DICT<EnvFeature, COMPLEX_VEC_SP>>();
    }
    return std::make_tuple(baseStart * TIME_ATOL, shiftT, this->complexCache[envName][baseStart]);
}

SP<DICT<EnvFeature, REAL_VEC_SP>> WaveCache::getRealCache(std::string envName)
{
    if (this->realCache.count(envName) == 0)
    {
        this->realCache[envName] = DICT<LLINT, SP<DICT<EnvFeature, REAL_VEC_SP>>>();
        this->realCache[envName][0] = std::make_shared<DICT<EnvFeature, REAL_VEC_SP>>();
    }
    return this->realCache[envName][0];
}

SP<DICT<EnvFeature, COMPLEX_VEC_SP>> WaveCache::getComplexCache(std::string envName)
{
    if (this->complexCache.count(envName) == 0)
    {
        this->complexCache[envName] = DICT<LLINT, SP<DICT<EnvFeature, COMPLEX_VEC_SP>>>();
        this->complexCache[envName][0] = std::make_shared<DICT<EnvFeature, COMPLEX_VEC_SP>>();
    }
    return this->complexCache[envName][0];
}

// Implementation of AbstractEnvelope
SP<AbstractEnvelope> AbstractEnvelope::operator+(const double &zero)
{
    if (zero != 0)
        throw std::invalid_argument("Cannot add a non-zero constant to an envelope");
    return shared_from_this();
}

SP<AbstractEnvelope> AbstractEnvelope::operator+(SP<AbstractEnvelope> &env)
{
    SP<EnvSum> result = std::make_shared<EnvSum>();
    SP<EnvSum> envSum = std::dynamic_pointer_cast<EnvSum>(env);
    if (envSum)
    {
        result->envs.reserve(envSum->envs.size() + 1);
        result->envs.push_back(shared_from_this());
        result->envs.insert(result->envs.end(), envSum->envs.begin(), envSum->envs.end());
    }
    else
    {
        result->envs.reserve(2);
        result->envs.push_back(shared_from_this());
        result->envs.push_back(env);
    }
    return result;
}

SP<AbstractEnvelope> AbstractEnvelope::operator+=(SP<AbstractEnvelope> &env)
{
    throw std::runtime_error("This situation is not allowed");
}

SP<AbstractEnvelope> AbstractEnvelope::operator*(const REAL &scale)
{
    return std::make_shared<EnvProd<REAL>>(shared_from_this(), scale);
}

SP<AbstractEnvelope> AbstractEnvelope::operator*(const COMPLEX &scale)
{
    return std::make_shared<EnvProd<COMPLEX>>(shared_from_this(), scale);
}

SP<AbstractEnvelope> AbstractEnvelope::operator>>(const double &dt)
{
    return std::make_shared<EnvShift>(shared_from_this(), dt);
}

SP<AbstractEnvelope> operator+(const double &zero, SP<AbstractEnvelope> env)
{
    if (zero != 0)
        throw std::invalid_argument("Cannot add a non-zero constant to an envelope");
    return env;
}

SP<AbstractEnvelope> operator*(const REAL &scale, SP<AbstractEnvelope> env)
{
    return (*env) * scale;
}

SP<AbstractEnvelope> operator*(const COMPLEX &scale, SP<AbstractEnvelope> env)
{
    return (*env) * scale;
}

// Implementation of Envelope

TP<double, SP<LazyArray<REAL>>> Envelope::realTimeFunc(WaveCache &wc, double dt)
{
    double start = this->start() + dt;
    auto [baseStart, shiftT, cache] = wc.getRealCache(this->toString(), start);
    EnvFeature feature = this->feature();
    if (cache->count(feature) == 0)
    {
        size_t sampleCount = ROUND(std::floor(this->duration() / wc.resolution_ + 0.5)) + END_PADDING;
        (*cache)[feature] = this->realModel(baseStart, sampleCount, wc.resolution_);
    }
    SP<LazyArray<REAL>> baseWave = std::make_shared<LazyArray<REAL>>((REAL)this->amp_, cache->at(feature));
    return std::make_tuple(shiftT, baseWave);
}

TP<double, SP<LazyArray<COMPLEX>>> Envelope::complexTimeFunc(WaveCache &wc, double dt)
{
    double start = this->start() + dt;
    auto [baseStart, shiftT, cache] = wc.getComplexCache(this->toString(), start);
    EnvFeature feature = this->feature();
    if (cache->count(feature) == 0)
    {
        size_t sampleCount = ROUND(std::floor(this->duration() / wc.resolution_ + 0.5)) + END_PADDING;
        (*cache)[feature] = this->complexModel(baseStart, sampleCount, wc.resolution_);
    }
    SP<LazyArray<COMPLEX>> baseWave = std::make_shared<LazyArray<COMPLEX>>((COMPLEX)this->amp_, cache->at(feature));
    return std::make_tuple(shiftT, baseWave);
}

// Implementation of EnvShift
SP<AbstractEnvelope> EnvShift::operator>>(const double &dt)
{
    return std::make_shared<EnvShift>(this->env, this->dt + dt);
}

TP<double, SP<LazyArray<REAL>>> EnvShift::realTimeFunc(WaveCache &cache, double dt)
{
    return this->env->realTimeFunc(cache, dt + this->dt);
}

TP<double, SP<LazyArray<COMPLEX>>> EnvShift::complexTimeFunc(WaveCache &cache, double dt)
{
    return this->env->complexTimeFunc(cache, dt + this->dt);
}

// Implementation of EnvProd
template <typename T>
bool EnvProd<T>::is_complex()
{
    return env->is_complex() || std::is_same_v<T, COMPLEX>;
}

template <typename T>
SP<AbstractEnvelope> EnvProd<T>::operator*(const REAL &scale)
{
    return std::make_shared<EnvProd<T>>(this->env, this->factor * scale);
}

template <typename T>
SP<AbstractEnvelope> EnvProd<T>::operator*(const COMPLEX &scale)
{
    return std::make_shared<EnvProd<COMPLEX>>(this->env, this->factor * scale);
}

template <typename T>
SP<AbstractEnvelope> EnvProd<T>::operator>>(const double &dt)
{
    return std::make_shared<EnvProd<T>>((*this->env) >> dt, this->factor);
}

template <typename T>
TP<double, SP<LazyArray<REAL>>> EnvProd<T>::realTimeFunc(WaveCache &wc, double dt)
{
    if constexpr (std::is_same_v<T, REAL>)
    {
        auto [t, wave] = this->env->realTimeFunc(wc, dt);
        *wave *= this->factor;
        return std::make_tuple(t, wave);
    }
    throw std::runtime_error("COMPLEX EnvProd should not call realTimeFunc!");
}

template <typename T>
TP<double, SP<LazyArray<COMPLEX>>> EnvProd<T>::complexTimeFunc(WaveCache &wc, double dt)
{
    auto [t, wave] = this->env->complexTimeFunc(wc, dt);
    *wave *= this->factor;
    return std::make_tuple(t, wave);
}

// Implementation of EnvSum
double EnvSum::start()
{
    return (*std::min_element(envs.begin(), envs.end(),
                              [](SP<AbstractEnvelope> &lenv, SP<AbstractEnvelope> &renv)
                              {
                                  return lenv->start() < renv->start();
                              }))
        ->start();
}

double EnvSum::end()
{
    return (*std::max_element(envs.begin(), envs.end(),
                              [](SP<AbstractEnvelope> &lenv, SP<AbstractEnvelope> &renv)
                              {
                                  return lenv->end() < renv->end();
                              }))
        ->end();
}

bool EnvSum::is_complex()
{
    return std::any_of(envs.begin(), envs.end(),
                       [](SP<AbstractEnvelope> &env)
                       {
                           return env->is_complex();
                       });
}

std::string EnvSum::toString() const
{
    std::string result = "EnvSum(";
    for (size_t i = 0; i < envs.size(); ++i)
    {
        result += envs[i]->toString();
        if (i < envs.size() - 1)
            result += ", ";
    }
    result += ")";
    return result;
}

SP<AbstractEnvelope> EnvSum::operator+(SP<AbstractEnvelope> &env)
{
    SP<EnvSum> result = std::make_shared<EnvSum>();
    SP<EnvSum> envSum = std::dynamic_pointer_cast<EnvSum>(env);
    if (envSum)
    {
        result->envs.reserve(envSum->envs.size() + this->envs.size());
        result->envs.insert(result->envs.end(), this->envs.begin(), this->envs.end());
        result->envs.insert(result->envs.end(), envSum->envs.begin(), envSum->envs.end());
    }
    else
    {
        result->envs.reserve(this->envs.size() + 1);
        result->envs.insert(result->envs.end(), this->envs.begin(), this->envs.end());
        result->envs.push_back(env);
    }
    return result;
}

SP<AbstractEnvelope> EnvSum::operator+=(SP<AbstractEnvelope> &env)
{
    SP<EnvSum> envSum = std::dynamic_pointer_cast<EnvSum>(env);
    if (envSum)
    {
        this->envs.insert(this->envs.end(), envSum->envs.begin(), envSum->envs.end());
    }
    else
    {
        this->envs.push_back(env);
    }
    return shared_from_this();
}

SP<AbstractEnvelope> EnvSum::operator*(const REAL &other)
{
    SP<EnvSum> result = std::make_shared<EnvSum>();
    result->envs.reserve(this->envs.size());
    for (size_t i = 0; i < this->envs.size(); i++)
    {
        result->envs.push_back((*this->envs[i]) * other);
    }
    return result;
}

SP<AbstractEnvelope> EnvSum::operator*(const COMPLEX &other)
{
    SP<EnvSum> result = std::make_shared<EnvSum>();
    result->envs.reserve(this->envs.size());
    for (size_t i = 0; i < this->envs.size(); i++)
    {
        result->envs.push_back((*this->envs[i]) * other);
    }
    return result;
}

SP<AbstractEnvelope> EnvSum::operator>>(const double &dt)
{
    SP<EnvSum> result = std::make_shared<EnvSum>();
    result->envs.reserve(this->envs.size());
    for (size_t i = 0; i < this->envs.size(); i++)
    {
        result->envs.push_back((*this->envs[i]) >> dt);
    }
    return result;
}

TP<double, SP<LazyArray<REAL>>> EnvSum::realTimeFunc(WaveCache &wc, double dt)
{
    size_t sizeOfEnvs = this->envs.size();
    VEC<double> startOfWaves(sizeOfEnvs);
    VEC<double> endOfWaves(sizeOfEnvs);
    VEC<SP<LazyArray<REAL>>> waves(sizeOfEnvs);

    for (size_t i = 0; i < sizeOfEnvs; i++)
    {
        auto [t_, wave_] = this->envs[i]->realTimeFunc(wc, dt);
        startOfWaves[i] = t_;
        endOfWaves[i] = t_ + wave_->size() * wc.resolution_;
        waves[i] = wave_;
    }
    double start = *std::min_element(startOfWaves.begin(), startOfWaves.end());
    double end = *std::max_element(endOfWaves.begin(), endOfWaves.end());
    size_t sampleCount = ROUND((end - start) / wc.resolution_);
    REAL_VEC_SP wave = std::make_shared<REAL_VEC>(sampleCount);
    for (size_t i = 0; i < sizeOfEnvs; i++)
    {
        size_t startIdx = ROUND((startOfWaves[i] - start) / wc.resolution_);
        for (size_t j = 0; j < waves[i]->size(); j++)
        {
            (*wave)[startIdx + j] += (*waves[i])[j];
        }
    }
    return std::make_tuple(start, std::make_shared<LazyArray<REAL>>((REAL)1.0, wave));
}

TP<double, SP<LazyArray<COMPLEX>>> EnvSum::complexTimeFunc(WaveCache &wc, double dt)
{
    size_t sizeOfEnvs = this->envs.size();
    VEC<double> startOfWaves(sizeOfEnvs);
    VEC<double> endOfWaves(sizeOfEnvs);
    VEC<SP<LazyArray<COMPLEX>>> waves(sizeOfEnvs);

    for (size_t i = 0; i < sizeOfEnvs; i++)
    {
        auto [t_, wave_] = this->envs[i]->complexTimeFunc(wc, dt);
        startOfWaves[i] = t_;
        endOfWaves[i] = t_ + wave_->size() * wc.resolution_;
        waves[i] = wave_;
    }
    double start = *std::min_element(startOfWaves.begin(), startOfWaves.end());
    double end = *std::max_element(endOfWaves.begin(), endOfWaves.end());
    size_t sampleCount = ROUND((end - start) / wc.resolution_);
    COMPLEX_VEC_SP wave = std::make_shared<COMPLEX_VEC>(sampleCount);
    for (size_t i = 0; i < sizeOfEnvs; i++)
    {
        size_t startIdx = ROUND((startOfWaves[i] - start) / wc.resolution_);
        for (size_t j = 0; j < waves[i]->size(); j++)
        {
            (*wave)[startIdx + j] += (*waves[i])[j];
        }
    }
    return std::make_tuple(start, std::make_shared<LazyArray<COMPLEX>>((COMPLEX)1.0, wave));
}

// Implementation of decodeEnvelope
TP<double, ANY_VEC_SP, bool> decodeEnvelope(SP<AbstractEnvelope> &env,
                                            WaveCache &wc)
{
    bool is_complex = env->is_complex();
    if (is_complex)
    {
        auto [envStart_, waveLazy_] = env->complexTimeFunc(wc, 0.0);
        auto wave_ = (env->type_ == EnvType::ENVSUM) ? waveLazy_->array_ : waveLazy_->eval(); // Always invokes eval() also works. But directly acquires the array_ is more efficient.
        return std::make_tuple(envStart_, wave_, is_complex);
    }
    else
    {
        auto [envStart_, waveLazy_] = env->realTimeFunc(wc, 0.0);
        auto wave_ = (env->type_ == EnvType::ENVSUM) ? waveLazy_->array_ : waveLazy_->eval(); // Always invokes eval() also works. But directly acquires the array_ is more efficient.
        return std::make_tuple(envStart_, wave_, is_complex);
    }
}

template <typename T>
double tailorEnvWave(const double &envStart,
                     SP<VEC<T>> &envWave,
                     double start,
                     double end,
                     WaveCache &wc)
{
    double envEnd = envStart + (envWave->size() - 1) * wc.resolution_;
    start = std::round(start / TIME_ATOL) / wc.resolutionOverAtol_ * wc.resolution_;
    if (start < envStart)
    {
        size_t prepadSampleNum = ROUND((envStart - start) / wc.resolution_);
        envWave->insert(envWave->begin(), prepadSampleNum, 0.0);
    }
    else
    {
        if (start > envEnd)
        {
            throw std::runtime_error("Input start > env.end()!");
        }
        else
        {
            LLINT startIdx = ROUND((start - envStart) / wc.resolution_);
            envWave->erase(envWave->begin(), envWave->begin() + startIdx);
        }
    }
    end = (std::round(end / TIME_ATOL) / wc.resolutionOverAtol_ - 1) * wc.resolution_;
    if (end > envEnd)
    {
        size_t postpadSampleNum = ROUND((end - envEnd) / wc.resolution_);
        envWave->insert(envWave->end(), postpadSampleNum, 0.0);
    }
    else
    {
        if (end < envStart)
        {
            throw std::runtime_error("Input end < env.start()!");
        }
        else
        {
            LLINT endIdx = ROUND((envEnd - end) / wc.resolution_);
            envWave->erase(envWave->end() - endIdx, envWave->end());
        }
    }
    return start;
}

TP<double, ANY_VEC_SP, bool> decodeEnvelope(SP<AbstractEnvelope> &env,
                                            WaveCache &wc,
                                            double start,
                                            double end)
{
    bool is_complex = env->is_complex();
    if (is_complex)
    {
        auto [envStart_, waveLazy_] = env->complexTimeFunc(wc, 0.0);
        auto wave_ = (env->type_ == EnvType::ENVSUM) ? waveLazy_->array_ : waveLazy_->eval(); // Always invokes eval() also works. But directly acquires the array_ is more efficient.
        start = tailorEnvWave<COMPLEX>(envStart_, wave_, start, end, wc);
        return std::make_tuple(start, wave_, is_complex);
    }
    else
    {
        auto [envStart_, waveLazy_] = env->realTimeFunc(wc, 0.0);
        auto wave_ = (env->type_ == EnvType::ENVSUM) ? waveLazy_->array_ : waveLazy_->eval(); // Always invokes eval() also works. But directly acquires the array_ is more efficient.
        start = tailorEnvWave<REAL>(envStart_, wave_, start, end, wc);
        return std::make_tuple(start, wave_, is_complex);
    }
}

/*
Function serializationHelper will be recursively called to serialize the envelope.
Each envPtr corresponds to a tuple(serializedData, envPtrs). And the envPtrs is a ptr (or a list of pointers if it is EnvSum) to other envs.
 */
void serializationHelper(void *ptr, VEC<TP<std::string *, void *>> &serializationStates, DICT<void *, SVIT> &ptrMap)
{
    if (ptr != nullptr && ptrMap.find(ptr) == ptrMap.end())
    {
        SVIT index = (SVIT)serializationStates.size();
        ptrMap[ptr] = index;
        TP<std::string *, void *> state = ((AbstractEnvelope *)ptr)->serializationState();
        serializationStates.push_back(state);
        EnvType et = *(EnvType *)&(*std::get<0>(state))[0];
        switch (et)
        {
        case EnvType::ENVMIX:
        {
            VEC<void *> *ptrs_ = (VEC<void *> *)std::get<1>(state);
            for (auto &ptr_ : *ptrs_)
            {
                serializationHelper(ptr_, serializationStates, ptrMap);
            }
            break;
        }
        case EnvType::ENVSUM:
        {
            VEC<void *> *ptrs_ = (VEC<void *> *)std::get<1>(state);
            for (auto &ptr_ : *ptrs_)
            {
                serializationHelper(ptr_, serializationStates, ptrMap);
            }
            break;
        }
        default:
            serializationHelper(std::get<1>(state), serializationStates, ptrMap);
            break;
        }
    }
}

SP<std::string> serialization(SP<AbstractEnvelope> &env)
{
    VEC<TP<std::string *, void *>> serializationStates;
    DICT<void *, SVIT> ptrMap;
    serializationHelper(env.get(), serializationStates, ptrMap);
    if (!std::is_same_v<SVIT, size_t> && serializationStates.size() >= 4294967296) // 2^32
    {
        throw std::runtime_error("Serialization might fail. Check the defined type of SVIT.");
    }

    std::string serializedData;
    std::string serializedPtr; // ptr will be convert to SVIT by ptrMap which indicated index in serializationStates

    for (auto &state : serializationStates)
    {
        std::string *serializedData_ = std::get<0>(state);

        serializedData += *serializedData_;

        EnvType et = *(EnvType *)&(*serializedData_)[0];
        switch (et)
        {
        case EnvType::ENVMIX:
        {
            VEC<void *> *ptrs = (VEC<void *> *)std::get<1>(state);
            for (auto &ptr : *ptrs)
            {
                SVIT index = ptrMap[ptr];
                serializedPtr += std::string((char *)&index, sizeof(SVIT));
            }
            delete ptrs;
            break;
        }
        case EnvType::ENVSUM:
        {
            VEC<void *> *ptrs = (VEC<void *> *)std::get<1>(state);
            for (auto &ptr : *ptrs)
            {
                SVIT index = ptrMap[ptr];
                serializedPtr += std::string((char *)&index, sizeof(SVIT));
            }
            delete ptrs;
            break;
        }
        default:
            void *ptr = std::get<1>(state);
            if (ptr != nullptr)
            {
                SVIT index = ptrMap[ptr];
                serializedPtr += std::string((char *)&index, sizeof(SVIT));
            }
            break;
        }
        delete serializedData_;
    }

    size_t size_ = serializationStates.size();
    SP<std::string> serialized = std::make_shared<std::string>((char *)&size_, sizeof(size_t));
    (*serialized) += serializedData;
    (*serialized) += serializedPtr;

    return serialized;
}

SP<AbstractEnvelope> deserialization(std::string &serialized)
{
    size_t serializationStateSize = *(size_t *)&(serialized[0]);
    VEC<SP<AbstractEnvelope>> deserializedEnvs(serializationStateSize);
    size_t current = sizeof(size_t);
    for (auto &env : deserializedEnvs)
    {
        EnvType et = *(EnvType *)&(serialized[current]);
        current += sizeof(EnvType);
        switch (et)
        {
        case EnvType::GAUSSIAN:
            env = std::make_shared<Gaussian>(serialized, current);
            break;
        case EnvType::GAUSSIANDRAG:
            env = std::make_shared<GaussianDRAG>(serialized, current);
            break;
        case EnvType::COSINEDRAG:
            env = std::make_shared<CosineDRAG>(serialized, current);
            break;
        case EnvType::TRIANGLE:
            env = std::make_shared<Triangle>(serialized, current);
            break;
        case EnvType::RECT:
            env = std::make_shared<Rect>(serialized, current);
            break;
        case EnvType::FLATTOP:
            env = std::make_shared<Flattop>(serialized, current);
            break;
        case EnvType::RIPPLERECT:
            env = std::make_shared<RippleRect>(serialized, current);
            break;
        case EnvType::MIXEXP:
            env = std::make_shared<MixExp>(serialized, current);
            break;
        case EnvType::ENVMIX:
            env = std::make_shared<EnvMix>(serialized, current);
            break;
        case EnvType::ENVSHIFT:
            env = std::make_shared<EnvShift>(serialized, current);
            break;
        case EnvType::ENVPROD:
        {
            bool is_complex = *(bool *)&(serialized[current]);
            current += sizeof(bool);
            if (is_complex)
            {
                env = std::make_shared<EnvProd<COMPLEX>>(serialized, current);
            }
            else
            {
                env = std::make_shared<EnvProd<REAL>>(serialized, current);
            }
            break;
        }
        case EnvType::ENVSUM:
            env = std::make_shared<EnvSum>(serialized, current);
            break;
        default:
            throw std::runtime_error("Unknown envelope type.");
            break;
        }
    }
    for (auto &env : deserializedEnvs)
    {
        env->setDeserializedEnvPtr(serialized, current, deserializedEnvs);
    }
    return deserializedEnvs[0];
}
