#include <afFramework.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace ambf;
using namespace chai3d;

namespace {

constexpr double kEpsilon = 1e-8;
constexpr unsigned int kSolverIterations = 4;
constexpr double kGuideErp = 0.1;
constexpr double kMaxGuideBiasVelocity = 0.1;
constexpr double kWireRadius = 0.0064;
constexpr double kMaxNormalImpulsePerUpdate = 0.1;
constexpr double kRotationContactSlop = 0.1;

struct RingWireGuidePoint
{
    cVector3d pointOnWire;
    cVector3d pointOnWireLocal;
    cVector3d tangent;
};

struct TorusWireContact
{
    cVector3d pointOnWire;
    cVector3d pointOnWireLocal;
    cVector3d pointOnRing;
    cVector3d normal;
    double penetration;
};

double distance(const cVector3d& first, const cVector3d& second)
{
    return (first - second).length();
}

cVector3d getMeshCentroid(cMultiMesh* mesh)
{
    cVector3d centroid(0.0, 0.0, 0.0);
    if (mesh == nullptr || mesh->getNumVertices() == 0) {
        return centroid;
    }

    for (unsigned int i = 0; i < mesh->getNumVertices(); ++i) {
        centroid += mesh->getVertexPos(i);
    }
    centroid *= 1.0 / static_cast<double>(mesh->getNumVertices());
    return centroid;
}

std::vector<cVector3d> resamplePolyline(const std::vector<cVector3d>& points, unsigned int sampleCount)
{
    if (points.size() < 2 || sampleCount < 2) {
        return points;
    }

    std::vector<double> cumulative{0.0};
    cumulative.reserve(points.size());
    for (size_t i = 1; i < points.size(); ++i) {
        cumulative.push_back(cumulative.back() + distance(points[i - 1], points[i]));
    }
    if (cumulative.back() <= kEpsilon) {
        return {};
    }

    std::vector<cVector3d> samples;
    samples.reserve(sampleCount);
    size_t segment = 0;
    for (unsigned int i = 0; i < sampleCount; ++i) {
        const double target = cumulative.back() * static_cast<double>(i) / static_cast<double>(sampleCount - 1);
        while (segment + 2 < points.size() && cumulative[segment + 1] < target) {
            ++segment;
        }
        const double span = cumulative[segment + 1] - cumulative[segment];
        const double fraction = span > kEpsilon ? (target - cumulative[segment]) / span : 0.0;
        samples.push_back(points[segment] + (points[segment + 1] - points[segment]) * fraction);
    }
    return samples;
}

std::vector<cVector3d> extractCenterline(cMultiMesh* mesh)
{
    if (mesh == nullptr || mesh->m_vtxIdxMap.empty()) {
        return {};
    }

    std::vector<std::vector<cVector3d>> candidates;
    candidates.reserve(mesh->m_vtxIdxMap.size());
    for (const auto& [originalIndex, importedIndices] : mesh->m_vtxIdxMap) {
        (void)originalIndex;
        std::vector<cVector3d> originalVertexCandidates;
        for (const int importedIndex : importedIndices) {
            if (importedIndex < 0) {
                return {};
            }
            for (int meshIndex = 0; meshIndex < mesh->getNumMeshes(); ++meshIndex) {
                cMesh* childMesh = mesh->getMesh(meshIndex);
                if (childMesh == nullptr ||
                    static_cast<unsigned int>(importedIndex) >= childMesh->getNumVertices()) {
                    continue;
                }
                const cVector3d candidate = childMesh->m_vertices->getLocalPos(importedIndex);
                const bool isDuplicate = std::any_of(
                    originalVertexCandidates.begin(),
                    originalVertexCandidates.end(),
                    [&candidate](const cVector3d& existing) {
                        return distance(existing, candidate) <= kEpsilon;
                    });
                if (!isDuplicate) {
                    originalVertexCandidates.push_back(candidate);
                }
            }
        }
        if (originalVertexCandidates.empty()) {
            return {};
        }
        candidates.push_back(std::move(originalVertexCandidates));
    }

    std::vector<std::vector<double>> costs(candidates.size());
    std::vector<std::vector<size_t>> predecessors(candidates.size());
    costs.front().assign(candidates.front().size(), 0.0);
    for (size_t i = 1; i < candidates.size(); ++i) {
        costs[i].assign(candidates[i].size(), std::numeric_limits<double>::infinity());
        predecessors[i].resize(candidates[i].size());
        for (size_t current = 0; current < candidates[i].size(); ++current) {
            for (size_t previous = 0; previous < candidates[i - 1].size(); ++previous) {
                const double cost = costs[i - 1][previous] +
                                    distance(candidates[i - 1][previous], candidates[i][current]);
                if (cost < costs[i][current]) {
                    costs[i][current] = cost;
                    predecessors[i][current] = previous;
                }
            }
        }
    }

    size_t selected = std::distance(
        costs.back().begin(),
        std::min_element(costs.back().begin(), costs.back().end()));
    std::vector<cVector3d> vertices(candidates.size());
    for (size_t i = candidates.size(); i-- > 0;) {
        vertices[i] = candidates[i][selected];
        if (i > 0) {
            selected = predecessors[i][selected];
        }
    }
    if (vertices.size() < 6) {
        return {};
    }

    std::vector<double> consecutiveDistances;
    consecutiveDistances.reserve(vertices.size() - 1);
    for (size_t i = 1; i < vertices.size(); ++i) {
        consecutiveDistances.push_back(distance(vertices[i - 1], vertices[i]));
    }
    std::vector<double> sortedDistances = consecutiveDistances;
    std::sort(sortedDistances.begin(), sortedDistances.end());
    const double typicalEdgeLength = sortedDistances[sortedDistances.size() / 2];
    if (typicalEdgeLength <= kEpsilon) {
        return {};
    }

    const double ringBreakThreshold = typicalEdgeLength * 4.0;
    size_t verticesPerRing = 0;
    for (size_t i = 0; i < consecutiveDistances.size(); ++i) {
        if (consecutiveDistances[i] > ringBreakThreshold) {
            verticesPerRing = i + 1;
            break;
        }
    }
    if (verticesPerRing < 3 || vertices.size() / verticesPerRing < 2) {
        return {};
    }

    std::vector<cVector3d> centerline;
    centerline.reserve(vertices.size() / verticesPerRing);
    for (size_t first = 0; first + verticesPerRing <= vertices.size(); first += verticesPerRing) {
        cVector3d center(0.0, 0.0, 0.0);
        for (size_t i = 0; i < verticesPerRing; ++i) {
            center += vertices[first + i];
        }
        centerline.push_back(center * (1.0 / static_cast<double>(verticesPerRing)));
    }
    return resamplePolyline(centerline, 100);
}

}  // namespace

class afRingWireConstraintPlugin : public afModelPlugin
{
public:
    virtual int init(const afModelPtr a_afModel, const afModelAttribsPtr a_modelAttribs) override
    {
        m_modelPtr = a_afModel;
        a_attribs = a_modelAttribs;

        if (m_modelPtr == nullptr) {
            std::cerr << "[ring_wire_constraint_plugin] ERROR: model pointer is null" << std::endl;
            return afInitStatus::ERROR;
        }

        ringBody_ = m_modelPtr->getRigidBody("ring_visual", true);
        wireBody_ = m_modelPtr->getRigidBody("wire_visual", true);
        if (ringBody_ == nullptr || wireBody_ == nullptr) {
            std::cerr << "[ring_wire_constraint_plugin] ERROR: could not find ring_visual or wire_visual bodies" << std::endl;
            return afInitStatus::ERROR;
        }
        if (ringBody_->m_bulletRigidBody == nullptr || wireBody_->m_bulletRigidBody == nullptr) {
            std::cerr << "[ring_wire_constraint_plugin] ERROR: ring_visual or wire_visual Bullet body is unavailable" << std::endl;
            return afInitStatus::ERROR;
        }

        ringBody_->m_bulletRigidBody->setIgnoreCollisionCheck(wireBody_->m_bulletRigidBody, true);
        wireBody_->m_bulletRigidBody->setIgnoreCollisionCheck(ringBody_->m_bulletRigidBody, true);

        cMultiMesh* ringMesh = ringBody_->getVisualObject();
        if (ringMesh == nullptr || ringMesh->getNumVertices() == 0) {
            std::cerr << "[ring_wire_constraint_plugin] ERROR: ring_visual has no visual mesh" << std::endl;
            return afInitStatus::ERROR;
        }
        ringCenterLocal_ = getMeshCentroid(ringMesh);
        double ringOuterRadius = 0.0;
        for (unsigned int i = 0; i < ringMesh->getNumVertices(); ++i) {
            const cVector3d delta = ringMesh->getVertexPos(i) - ringCenterLocal_;
            const double radius = std::hypot(delta.x(), delta.y());
            ringInnerRadius_ = std::min(ringInnerRadius_, radius);
            ringOuterRadius = std::max(ringOuterRadius, radius);
        }
        ringMajorRadius_ = 0.5 * (ringInnerRadius_ + ringOuterRadius);
        ringTubeRadius_ = 0.5 * (ringOuterRadius - ringInnerRadius_);
        guideClearance_ = ringInnerRadius_ - kWireRadius;
        if (!std::isfinite(guideClearance_) || guideClearance_ <= kEpsilon ||
            ringTubeRadius_ <= kEpsilon) {
            std::cerr << "[ring_wire_constraint_plugin] ERROR: could not determine ring dimensions"
                      << std::endl;
            return afInitStatus::ERROR;
        }

        wireCenterlineLocal_ = extractCenterline(wireBody_->getVisualObject());
        if (wireCenterlineLocal_.size() < 2) {
            cMultiMesh* wireMesh = wireBody_->getVisualObject();
            std::cerr << "[ring_wire_constraint_plugin] ERROR: could not extract a centerline from wire_visual"
                      << " (meshes=" << (wireMesh == nullptr ? 0 : wireMesh->getNumMeshes())
                      << ", sourceVertexMap=" << (wireMesh == nullptr ? 0 : wireMesh->m_vtxIdxMap.size())
                      << ")" << std::endl;
            return afInitStatus::ERROR;
        }

        std::cerr << "[ring_wire_constraint_plugin] initialized clearance guide with "
                  << wireCenterlineLocal_.size() - 1 << " segments and clearance="
                  << guideClearance_ << std::endl;
        return afInitStatus::SUCCESS;
    }

    virtual void physicsUpdate(double dt) override
    {
        if (dt <= kEpsilon || ringBody_ == nullptr || wireBody_ == nullptr ||
            ringBody_->m_bulletRigidBody == nullptr || wireBody_->m_bulletRigidBody == nullptr) {
            return;
        }

        btRigidBody* ring = ringBody_->m_bulletRigidBody;
        btRigidBody* wire = wireBody_->m_bulletRigidBody;
        const cTransform ringTransform = ringBody_->getGlobalTransform();
        const cVector3d ringCenter = ringTransform * ringCenterLocal_;
        cVector3d ringAxis = ringTransform.getLocalRot() * cVector3d(0.0, 0.0, 1.0);
        if (ringAxis.lengthsq() <= kEpsilon) {
            return;
        }
        ringAxis.normalize();

        const cTransform wireTransform = wireBody_->getGlobalTransform();
        if (!hasPreviousWireTransform_) {
            previousWireTransform_ = wireTransform;
            hasPreviousWireTransform_ = true;
        }
        RingWireGuidePoint guidePoint;
        std::vector<TorusWireContact> rotationContacts;
        double closestDistanceSquared = std::numeric_limits<double>::infinity();
        for (size_t i = 1; i < wireCenterlineLocal_.size(); ++i) {
            const cVector3d firstLocal = wireCenterlineLocal_[i - 1];
            const cVector3d secondLocal = wireCenterlineLocal_[i];
            const cVector3d first = wireTransform * firstLocal;
            const cVector3d second = wireTransform * secondLocal;
            const cVector3d segment = second - first;
            const double segmentLengthSquared = segment.lengthsq();
            if (segmentLengthSquared <= kEpsilon) {
                continue;
            }
            const double fraction = std::clamp(
                (ringCenter - first).dot(segment) / segmentLengthSquared,
                0.0,
                1.0);
            const cVector3d candidateLocal = firstLocal + (secondLocal - firstLocal) * fraction;
            const cVector3d candidate = wireTransform * candidateLocal;
            const double candidateDistanceSquared = (ringCenter - candidate).lengthsq();
            if (candidateDistanceSquared < closestDistanceSquared) {
                closestDistanceSquared = candidateDistanceSquared;
                guidePoint = {
                    candidate,
                    candidateLocal,
                    segment * (1.0 / std::sqrt(segmentLengthSquared))
                };
            }

            const cVector3d ringToWire = candidate - ringCenter;
            const double axialDistance = ringToWire.dot(ringAxis);
            const cVector3d radial = ringToWire - ringAxis * axialDistance;
            const double radialDistance = radial.length();
            if (radialDistance <= kEpsilon) {
                continue;
            }

            const cVector3d torusCenter =
                ringCenter + radial * (ringMajorRadius_ / radialDistance);
            const cVector3d torusToWire = candidate - torusCenter;
            const double surfaceDistance = torusToWire.length();
            const double torusPenetration =
                ringTubeRadius_ + kWireRadius - surfaceDistance - kRotationContactSlop;
            if (torusPenetration <= 0.0 || surfaceDistance <= kEpsilon) {
                continue;
            }

            const cVector3d wireToTorus = torusToWire * (-1.0 / surfaceDistance);
            rotationContacts.push_back({
                candidate,
                candidateLocal,
                torusCenter - wireToTorus * ringTubeRadius_,
                wireToTorus,
                torusPenetration
            });
        }
        if (!std::isfinite(closestDistanceSquared)) {
            previousWireTransform_ = wireTransform;
            return;
        }

        const cVector3d wireToRing = ringCenter - guidePoint.pointOnWire;
        const double centerOffset = wireToRing.length();
        const bool hasGuideContact = centerOffset > guideClearance_;
        if (!hasGuideContact && rotationContacts.empty()) {
            previousWireTransform_ = wireTransform;
            return;
        }
        const cVector3d ringContactNormal = hasGuideContact
                                               ? wireToRing * (-1.0 / centerOffset)
                                               : cVector3d(0.0, 0.0, 0.0);
        const double penetration = hasGuideContact ? centerOffset - guideClearance_ : 0.0;

        double accumulatedImpulse = 0.0;
        for (unsigned int iteration = 0; hasGuideContact && iteration < kSolverIterations; ++iteration) {
            const btVector3 ringPoint(ringCenter.x(), ringCenter.y(), ringCenter.z());
            const btVector3 wirePoint(
                guidePoint.pointOnWire.x(),
                guidePoint.pointOnWire.y(),
                guidePoint.pointOnWire.z());
            const btVector3 normal(
                ringContactNormal.x(),
                ringContactNormal.y(),
                ringContactNormal.z());
            const btVector3 ringOffset = ringPoint - ring->getCenterOfMassPosition();
            const btVector3 wireOffset = wirePoint - wire->getCenterOfMassPosition();

            const btScalar inverseEffectiveMass =
                ring->computeImpulseDenominator(ringPoint, normal) +
                wire->computeImpulseDenominator(wirePoint, normal);
            if (inverseEffectiveMass <= SIMD_EPSILON) {
                continue;
            }

            btVector3 wireVelocity;
            if (wire->getInvMass() > SIMD_EPSILON) {
                wireVelocity = wire->getVelocityInLocalPoint(wireOffset);
            }
            else {
                const cVector3d previousWirePoint =
                    previousWireTransform_ * guidePoint.pointOnWireLocal;
                const cVector3d kinematicWireVelocity =
                    (guidePoint.pointOnWire - previousWirePoint) * (1.0 / dt);
                wireVelocity = btVector3(
                    kinematicWireVelocity.x(),
                    kinematicWireVelocity.y(),
                    kinematicWireVelocity.z());
            }
            const btScalar normalVelocity =
                normal.dot(ring->getVelocityInLocalPoint(ringOffset) - wireVelocity);
            const btScalar bias = std::min(
                static_cast<btScalar>(kMaxGuideBiasVelocity),
                static_cast<btScalar>(kGuideErp * penetration / dt));
            const btScalar unconstrainedImpulse =
                (std::max(bias, btScalar(0.0)) - normalVelocity) / inverseEffectiveMass;
            const btScalar impulseMagnitude = std::min(
                unconstrainedImpulse,
                static_cast<btScalar>(kMaxNormalImpulsePerUpdate - accumulatedImpulse));
            if (impulseMagnitude <= 0.0) {
                continue;
            }

            const btVector3 impulse = normal * impulseMagnitude;
            ring->applyImpulse(impulse, ringOffset);
            wire->applyImpulse(-impulse, wireOffset);
            accumulatedImpulse += impulseMagnitude;
        }

        for (unsigned int iteration = 0; iteration < kSolverIterations; ++iteration) {
            for (const TorusWireContact& contact : rotationContacts) {
                const btVector3 ringPoint(
                    contact.pointOnRing.x(),
                    contact.pointOnRing.y(),
                    contact.pointOnRing.z());
                const btVector3 wirePoint(
                    contact.pointOnWire.x(),
                    contact.pointOnWire.y(),
                    contact.pointOnWire.z());
                const btVector3 normal(
                    contact.normal.x(),
                    contact.normal.y(),
                    contact.normal.z());
                const btVector3 ringOffset = ringPoint - ring->getCenterOfMassPosition();
                const btVector3 wireOffset = wirePoint - wire->getCenterOfMassPosition();
                const btScalar inverseEffectiveMass =
                    ring->computeImpulseDenominator(ringPoint, normal) +
                    wire->computeImpulseDenominator(wirePoint, normal);
                if (inverseEffectiveMass <= SIMD_EPSILON) {
                    continue;
                }

                btVector3 wireVelocity;
                if (wire->getInvMass() > SIMD_EPSILON) {
                    wireVelocity = wire->getVelocityInLocalPoint(wireOffset);
                }
                else {
                    const cVector3d previousWirePoint =
                        previousWireTransform_ * contact.pointOnWireLocal;
                    const cVector3d kinematicWireVelocity =
                        (contact.pointOnWire - previousWirePoint) * (1.0 / dt);
                    wireVelocity = btVector3(
                        kinematicWireVelocity.x(),
                        kinematicWireVelocity.y(),
                        kinematicWireVelocity.z());
                }

                const btScalar normalVelocity =
                    normal.dot(ring->getVelocityInLocalPoint(ringOffset) - wireVelocity);
                const btScalar bias = std::min(
                    static_cast<btScalar>(kMaxGuideBiasVelocity),
                    static_cast<btScalar>(kGuideErp * contact.penetration / dt));
                const btScalar unconstrainedImpulse =
                    (std::max(bias, btScalar(0.0)) - normalVelocity) / inverseEffectiveMass;
                const btScalar impulseMagnitude = std::min(
                    unconstrainedImpulse,
                    static_cast<btScalar>(kMaxNormalImpulsePerUpdate - accumulatedImpulse));
                if (impulseMagnitude <= 0.0) {
                    continue;
                }

                const btVector3 impulse = normal * impulseMagnitude;
                ring->applyImpulse(impulse, ringOffset);
                wire->applyImpulse(-impulse, wireOffset);
                accumulatedImpulse += impulseMagnitude;
            }
        }

        previousWireTransform_ = wireTransform;
    }

    virtual bool close() override
    {
        if (ringBody_ != nullptr && wireBody_ != nullptr &&
            ringBody_->m_bulletRigidBody != nullptr && wireBody_->m_bulletRigidBody != nullptr) {
            ringBody_->m_bulletRigidBody->setIgnoreCollisionCheck(wireBody_->m_bulletRigidBody, false);
            wireBody_->m_bulletRigidBody->setIgnoreCollisionCheck(ringBody_->m_bulletRigidBody, false);
        }
        ringBody_ = nullptr;
        wireBody_ = nullptr;
        wireCenterlineLocal_.clear();
        hasPreviousWireTransform_ = false;
        return true;
    }

private:
    afRigidBodyPtr ringBody_ = nullptr;
    afRigidBodyPtr wireBody_ = nullptr;
    cVector3d ringCenterLocal_ = cVector3d(0.0, 0.0, 0.0);
    std::vector<cVector3d> wireCenterlineLocal_;
    double ringInnerRadius_ = std::numeric_limits<double>::infinity();
    double ringMajorRadius_ = 0.0;
    double ringTubeRadius_ = 0.0;
    double guideClearance_ = 0.0;
    cTransform previousWireTransform_;
    bool hasPreviousWireTransform_ = false;
};

AF_REGISTER_MODEL_PLUGIN(afRingWireConstraintPlugin)
