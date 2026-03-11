import database
import models

print("Provisioning database tables...")
models.Base.metadata.create_all(bind=database.engine)
print("Tables created successfully.")
