#include <afFramework.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace ambf;
using namespace chai3d;

namespace {

cVector3d getMeshCentroid(cMultiMesh* mesh)
{
    cVector3d centroid(0.0, 0.0, 0.0);
    if (mesh == nullptr) {
        return centroid;
    }

    const unsigned int numVertices = mesh->getNumVertices();
    if (numVertices == 0) {
        return centroid;
    }

    for (unsigned int i = 0; i < numVertices; ++i) {
        centroid += mesh->getVertexPos(i);
    }

    centroid *= 1.0 / static_cast<double>(numVertices);
    return centroid;
}

double getMeshRadius(cMultiMesh* mesh, const cVector3d& center)
{
    if (mesh == nullptr) {
        return 0.0;
    }

    const unsigned int numVertices = mesh->getNumVertices();
    if (numVertices == 0) {
        return 0.0;
    }

    double maxDistance = 0.0;
    for (unsigned int i = 0; i < numVertices; ++i) {
        const cVector3d delta = mesh->getVertexPos(i) - center;
        const double distance = delta.length();
        if (distance > maxDistance) {
            maxDistance = distance;
        }
    }

    return maxDistance;
}

}  // namespace

class afRingWireConstraintPlugin : public afModelPlugin
{
public:
    virtual int init(const afModelPtr a_afModel, const afModelAttribsPtr a_modelAttribs) override
    {
        m_modelPtr = a_afModel;
        this->a_attribs = a_modelAttribs;

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

        cMultiMesh* ringMesh = ringBody_->getVisualObject();
        if (ringMesh != nullptr) {
            ringCenterLocal_ = getMeshCentroid(ringMesh);
            ringRadius_ = getMeshRadius(ringMesh, ringCenterLocal_);
        }

        if (ringRadius_ <= 0.0) {
            ringRadius_ = 0.0185;
        }

        wireRadius_ = 0.0064;
        clampRadius_ = std::max(0.001, ringRadius_ * 0.5 - wireRadius_);

        std::cerr << "@@@INFO! " << __FILE__ << " [ring_wire_constraint_plugin] initialized with ringRadius=" << ringRadius_ << ", clampRadius=" << clampRadius_ << std::endl;
        return afInitStatus::SUCCESS;
    }

    virtual void physicsUpdate(double dt) override
    {
        (void)dt;

        if (ringBody_ == nullptr || wireBody_ == nullptr) {
            return;
        }

        if (ringBody_->m_bulletRigidBody == nullptr || wireBody_->m_bulletRigidBody == nullptr) {
            return;
        }

        cMultiMesh* wireMesh = wireBody_->getVisualObject();
        if (wireMesh == nullptr || wireMesh->getNumVertices() == 0) {
            return;
        }

        const cTransform ringTransform = ringBody_->getGlobalTransform();
        const cVector3d ringCenter = ringTransform.getLocalPos();
        const cMatrix3d ringRot = ringTransform.getLocalRot();
        cVector3d ringNormal = ringRot * cVector3d(0.0, 0.0, 1.0);
        ringNormal.normalize();

        const cTransform wireTransform = wireBody_->getGlobalTransform();
        const double innerRadius = std::max(0.001, ringRadius_ - wireRadius_);
        const double maxViolation = 0.03;

        for (unsigned int i = 0; i < wireMesh->getNumVertices(); ++i) {
            const cVector3d localPoint = wireMesh->getVertexPos(i);
            const cVector3d worldPoint = wireTransform * localPoint;
            const cVector3d delta = worldPoint - ringCenter;
            const double normalComponent = delta.dot(ringNormal);
            const cVector3d planar = delta - normalComponent * ringNormal;
            const double planarDistance = planar.length();

            if (planarDistance <= innerRadius + 1e-6) {
                continue;
            }

            const cVector3d planarDir = planar / std::max(planarDistance, 1e-8);
            const double violation = std::min(planarDistance - innerRadius, maxViolation);
            cerr << "[ring_wire_constraint_plugin] vertex " << i << " violation: " << violation << std::endl;
            const cVector3d impulseWorld = -planarDir * (violation * 6.0);
            const cVector3d relPos = (ringCenter + ringNormal * 0.01 + planarDir * std::min(ringRadius_ * 0.5, 0.02)) - ringCenter;

            const btVector3 impulseBt(impulseWorld.x(), impulseWorld.y(), impulseWorld.z());
            const btVector3 relPosBt(relPos.x(), relPos.y(), relPos.z());
            // ringBody_->m_bulletRigidBody->applyImpulse(impulseBt, relPosBt);
        }
    }

    virtual bool close() override
    {
        ringBody_ = nullptr;
        wireBody_ = nullptr;
        return true;
    }

private:
    afRigidBodyPtr ringBody_ = nullptr;
    afRigidBodyPtr wireBody_ = nullptr;
    cVector3d ringCenterLocal_ = cVector3d(0.0, 0.0, 0.0);
    double ringRadius_ = 0.0185;
    double wireRadius_ = 0.0064;
    double clampRadius_ = 0.0025;
};

AF_REGISTER_MODEL_PLUGIN(afRingWireConstraintPlugin)
